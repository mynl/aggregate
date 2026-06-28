r"""Variable-rating contract terms: the shared ``ContractTerms`` taxonomy.

A *contract term* is a deterministic, vectorized function :math:`\varphi` of a
realized loss quantity that fills **one** leg of a Gross / Ceded / Net P&L (the
legs model, ``dev/plan-variable-rating-appendix.md`` section 1). Pushed forward
over the loss distribution -- 1-D on an aggregate basis, 2-D on an occurrence
basis (appendix section 2) -- it makes that leg stochastic. Six features share
this shape: reinstatement premium
(:class:`aggregate.reinstatement.ReinstatementTerms`), retro, swing, slide,
profit commission and corridor.

This module holds the thin :class:`ContractTerms` base and the five Phase-3
single-leg features. The reinstatement specialization is a **two-map** feature
(a premium decorator ``h(R)`` *and* the annual-cap loss transform ``A(R)``) and
lives with its analysis engine in ``reinstatement.py``; it subclasses
:class:`ContractTerms` there.

Each feature owns its vectorized :meth:`~ContractTerms.phi`, the P&L leg it fills
(:attr:`~ContractTerms.target_leg`) and the loss quantity ``phi`` reads
(:attr:`~ContractTerms.loss_basis`, appendix section 4). The leg-ratio features
(slide / pc / corridor) read the **ceded loss ratio** ``LR``; the analysis layer
converts ``LR`` to / from currency with the ceded premium, so ``phi`` stays a
pure single-argument map here.

Submodule access only (no top-level re-export)::

    from aggregate.contract_terms import SwingTerms
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    'ContractTerms',
    'RetroTerms',
    'SwingTerms',
    'SlideTerms',
    'ProfitCommissionTerms',
    'CorridorTerms',
]

#: The P&L legs a contract term may fill (appendix section 1).
LEGS = ('gross_premium', 'ceded_premium', 'expense', 'ceded_loss')


class ContractTerms:
    """Base for a deterministic variable-rating term.

    A description of one feature's economics: the vectorized map :meth:`phi` from
    a realized loss quantity (named by :attr:`loss_basis`) to the value of the P&L
    leg it fills (named by :attr:`target_leg`). Subclasses are frozen dataclasses
    that set the two class-level metadata strings and implement :meth:`phi`; the
    base supplies the shared finite / nonnegative / monotone validation helper
    :meth:`_check_vectorized`.

    Attributes
    ----------
    target_leg : str
        Which P&L leg this term fills: one of :data:`LEGS`. Set by each subclass.
    loss_basis : str
        The loss quantity ``phi`` reads (appendix section 4), e.g. ``'ceded_loss'``,
        ``'ceded_lr'``, ``'net_account_loss'``, ``'occurrence_recovery'``. Set by
        each subclass.
    """

    #: subclasses override these two class-level constants
    target_leg = None
    loss_basis = None

    def phi(self, x):
        """Vectorized map from the loss basis to the target-leg value.

        Parameters
        ----------
        x : float or ndarray
            A realized value of the feature's :attr:`loss_basis`.

        Returns
        -------
        float or ndarray
            The deterministic leg value (or, for the ratio features, the
            transformed loss ratio the analysis layer scales by ceded premium).
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement phi().')

    # ------------------------------------------------------------------
    # shared validation utility (generalized from ReinstatementTerms)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_vectorized(fn, lo, hi, *, where, nonnegative=True,
                          nondecreasing=None, n=257):
        """Probe a vectorized scalar map on ``[lo, hi]`` for its invariants.

        Confirms ``fn`` accepts and returns an ``(n,)`` array of finite values and
        (optionally) is nonnegative and monotone. Raises :class:`ValueError`
        tagged with ``where`` on any failure; returns the probed values otherwise.

        Parameters
        ----------
        fn : callable
            Vectorized scalar map to probe.
        lo, hi : float
            Probe range endpoints.
        where : str
            Caller tag prefixing every error message.
        nonnegative : bool, default True
            Require ``fn >= 0`` over the probe.
        nondecreasing : bool or None, default None
            ``True`` requires a nondecreasing ``fn``; ``False`` a nonincreasing
            one; ``None`` skips the monotonicity check.
        n : int, default 257
            Number of probe points.
        """
        probe = np.linspace(float(lo), float(hi), n)
        try:
            vals = np.asarray(fn(probe), dtype=float)
        except Exception as e:                            # noqa: BLE001
            raise ValueError(
                f'{where}: must accept and return a NumPy array (it failed to '
                f'evaluate on a vector). Underlying error: {e!r}') from e
        if vals.shape != probe.shape:
            raise ValueError(
                f'{where}: must be vectorized (input shape {probe.shape}, output '
                f'shape {vals.shape}).')
        if not np.all(np.isfinite(vals)):
            raise ValueError(
                f'{where}: must be finite over [{float(lo):g}, {float(hi):g}].')
        if nonnegative and np.any(vals < -1e-9):
            raise ValueError(f'{where}: must be nonnegative.')
        if nondecreasing is True and np.any(np.diff(vals) < -1e-9):
            raise ValueError(f'{where}: must be nondecreasing.')
        if nondecreasing is False and np.any(np.diff(vals) > 1e-9):
            raise ValueError(f'{where}: must be nonincreasing.')
        return vals


# ======================================================================
# collared affine premium (retro and swing share the machinery)
# ======================================================================
@dataclass(frozen=True)
class _CollaredAffineTerms(ContractTerms):
    r"""Shared collared affine premium map ``clip(basic + lcm * x, min, max)``.

    The common shape of :class:`RetroTerms` (gross premium of net account loss)
    and :class:`SwingTerms` (ceded premium of ceded loss): an additive base
    ``basic``, a loss multiplier ``lcm``, and a ``[minimum, maximum]`` collar.
    Not used directly -- the two leaf classes set :attr:`target_leg` /
    :attr:`loss_basis`.

    Parameters
    ----------
    basic : float
        Additive base premium ``b`` (currency); finite and ``>= 0``.
    lcm : float
        Loss-conversion multiplier ``m`` applied to the loss; finite and ``>= 0``.
    minimum : float, optional
        Premium floor. Default (``None``) is ``basic`` (premium never below base).
    maximum : float, optional
        Premium cap. Default (``None``) is ``+inf`` (uncapped).

    Notes
    -----
    ``phi(x) = clip(basic + lcm * x, minimum, maximum)`` -- nondecreasing in ``x``,
    flat below ``(minimum - basic) / lcm`` and above ``(maximum - basic) / lcm``.
    """

    basic: float
    lcm: float
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self):
        cn = type(self).__name__
        if not (np.isfinite(self.basic) and self.basic >= 0):
            raise ValueError(
                f'{cn}: basic must be finite and >= 0, got {self.basic!r}.')
        if not (np.isfinite(self.lcm) and self.lcm >= 0):
            raise ValueError(
                f'{cn}: lcm must be finite and >= 0, got {self.lcm!r}.')
        lo = float(self.basic) if self.minimum is None else float(self.minimum)
        hi = np.inf if self.maximum is None else float(self.maximum)
        object.__setattr__(self, 'minimum', lo)
        object.__setattr__(self, 'maximum', hi)
        if not (np.isfinite(lo) and lo >= 0):
            raise ValueError(
                f'{cn}: minimum must be finite and >= 0, got {lo!r}.')
        if not (lo <= hi):
            raise ValueError(
                f'{cn}: minimum ({lo:g}) must be <= maximum ({hi:g}).')

    def phi(self, x):
        """Collared affine premium ``clip(basic + lcm * x, minimum, maximum)``."""
        x = np.asarray(x, dtype=float)
        return np.clip(self.basic + self.lcm * x, self.minimum, self.maximum)


class RetroTerms(_CollaredAffineTerms):
    r"""Retrospective rating: gross premium as a collared affine map of loss.

    Account-level **rating clause** (not a reinsurance decorator, decision 3): the
    gross premium is ``clip(basic + lcm * L, minimum, maximum)`` where ``L`` is the
    net account loss (after any inuring reinsurance). Fills the gross-premium leg.

    See :class:`_CollaredAffineTerms` for the parameters.

    Examples
    --------
    ``basic=1000, lcm=1.10, minimum=1000, maximum=2500`` (worked example): ``L=0``
    -> ``1000``; ``L=500`` -> ``1550``; ``L=1500`` -> ``2500`` (capped).

    >>> RetroTerms(basic=1000.0, lcm=1.10, minimum=1000.0, maximum=2500.0).phi(
    ...     np.array([0.0, 500.0, 1500.0]))
    array([1000., 1550., 2500.])
    """

    target_leg = 'gross_premium'
    loss_basis = 'net_account_loss'


class SwingTerms(_CollaredAffineTerms):
    r"""Swing-rated reinsurance: ceded premium as a collared affine map of loss.

    Decorates a reinsurance layer's **premium** slot (replacing ``deposit | rol |
    rate``, appendix section 3): the ceded premium is ``clip(basic + lcm * A,
    minimum, maximum)`` where ``A`` is the ceded loss to the layer. Fills the
    ceded-premium leg.

    See :class:`_CollaredAffineTerms` for the parameters.

    Examples
    --------
    ``basic=0, lcm=1.0, minimum=100, maximum=300`` (pay-back-losses to a max,
    worked example): ``A=50`` -> ``100``; ``A=200`` -> ``200``; ``A=400`` ->
    ``300``.

    >>> SwingTerms(basic=0.0, lcm=1.0, minimum=100.0, maximum=300.0).phi(
    ...     np.array([50.0, 200.0, 400.0]))
    array([100., 200., 300.])
    """

    target_leg = 'ceded_premium'
    loss_basis = 'ceded_loss'


# ======================================================================
# sliding-scale ceding commission
# ======================================================================
@dataclass(frozen=True)
class SlideTerms(ContractTerms):
    r"""Sliding-scale ceding commission: a decreasing PWL function of ceded LR.

    The commission rate slides down as the ceded loss ratio rises, from
    ``(commission, loss_ratio)`` anchor pairs (decision 2), piecewise-linear
    between anchors and **flat outside the end anchors** (the implicit min / max
    commission). Decorates a layer's **commission** slot (replacing ``cede``).
    Fills the **expense** leg as a credit: commission currency ``= phi(LR) *
    ceded_premium`` (the analysis layer multiplies by premium).

    Parameters
    ----------
    anchors : tuple of (float, float)
        ``(commission, loss_ratio)`` pairs, e.g. ``((0.45, 0.60), (0.25, 0.70),
        (0.19, 0.80))``. Sorted internally by loss ratio; commissions must be
        nonincreasing as the loss ratio increases (a sliding scale) and the loss
        ratios must be distinct.

    Notes
    -----
    ``phi(LR)`` interpolates commission against loss ratio with flat extrapolation
    (:func:`numpy.interp` clamps to the end anchors).

    Examples
    --------
    Anchors ``((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))`` (worked example):
    ``LR=0.55`` -> ``0.45`` (flat); ``LR=0.65`` -> ``0.35``; ``LR=0.75`` ->
    ``0.22``; ``LR=0.90`` -> ``0.19`` (flat).

    >>> SlideTerms.from_anchors((0.45, 0.60), (0.25, 0.70), (0.19, 0.80)).phi(
    ...     np.array([0.55, 0.65, 0.75, 0.90]))
    array([0.45, 0.35, 0.22, 0.19])
    """

    target_leg = 'expense'
    loss_basis = 'ceded_lr'

    anchors: tuple

    @classmethod
    def from_anchors(cls, *pairs):
        """Build from ``(commission, loss_ratio)`` pairs given as positional args."""
        return cls(anchors=tuple(pairs))

    def __post_init__(self):
        pairs = tuple((float(c), float(lr)) for (c, lr) in self.anchors)
        if len(pairs) < 1:
            raise ValueError(
                'SlideTerms: at least one (commission, loss_ratio) anchor '
                'required.')
        pairs = tuple(sorted(pairs, key=lambda p: p[1]))   # by loss ratio
        object.__setattr__(self, 'anchors', pairs)
        comm = [c for c, _ in pairs]
        lr = [l for _, l in pairs]
        if any((not np.isfinite(c)) or c < 0 for c in comm):
            raise ValueError(
                f'SlideTerms: commissions must be finite and >= 0, got {comm!r}.')
        if any((not np.isfinite(l)) or l < 0 for l in lr):
            raise ValueError(
                f'SlideTerms: loss ratios must be finite and >= 0, got {lr!r}.')
        if len(set(lr)) != len(lr):
            raise ValueError(
                f'SlideTerms: loss-ratio anchors must be distinct, got {lr!r}.')
        if any(comm[i + 1] > comm[i] + 1e-12 for i in range(len(comm) - 1)):
            raise ValueError(
                f'SlideTerms: commission must be nonincreasing in loss ratio '
                f'(a sliding scale), got {pairs!r}.')

    @property
    def _loss_ratios(self):
        """Anchor loss ratios (the ascending ``xp`` for interpolation)."""
        return np.array([lr for _, lr in self.anchors], dtype=float)

    @property
    def _commissions(self):
        """Anchor commissions (the ``fp`` for interpolation)."""
        return np.array([c for c, _ in self.anchors], dtype=float)

    def phi(self, loss_ratio):
        """Commission fraction at ``loss_ratio`` (flat outside the end anchors)."""
        lr = np.asarray(loss_ratio, dtype=float)
        # numpy.interp clamps to fp[0] / fp[-1] beyond xp -> flat extrapolation
        return np.interp(lr, self._loss_ratios, self._commissions)


# ======================================================================
# profit commission
# ======================================================================
@dataclass(frozen=True)
class ProfitCommissionTerms(ContractTerms):
    r"""Profit commission: a share of underwriting profit returned to the cedant.

    ``PC = share * (1 - LR - allowance)_+`` as a fraction of ceded premium, where
    ``LR`` is the ceded loss ratio and ``allowance`` the reinsurer's expense
    allowance (margin retained before profit is shared). DecL ``pc <share> after
    <allowance>``. Fills the **expense** leg as a credit: ``PC currency = phi(LR) *
    ceded_premium`` (the analysis layer multiplies by premium).

    Parameters
    ----------
    share : float
        Profit share ``p`` in ``[0, 1]``.
    allowance : float, default 0.0
        Reinsurer expense allowance (loss-ratio points); finite and ``>= 0``.

    Notes
    -----
    ``phi(LR) = share * max(1 - LR - allowance, 0)`` -- nonincreasing in ``LR``,
    zero once ``LR >= 1 - allowance``.

    Examples
    --------
    ``share=0.25, allowance=0.10`` (worked example): ``LR=0.50`` -> ``0.10``;
    ``LR=0.60`` -> ``0.075``; ``LR=0.95`` -> ``0`` (clamped).

    >>> ProfitCommissionTerms(share=0.25, allowance=0.10).phi(
    ...     np.array([0.50, 0.60, 0.95]))
    array([0.1  , 0.075, 0.   ])
    """

    target_leg = 'expense'
    loss_basis = 'ceded_lr'

    share: float
    allowance: float = 0.0

    def __post_init__(self):
        if not (np.isfinite(self.share) and 0.0 <= self.share <= 1.0):
            raise ValueError(
                f'ProfitCommissionTerms: share must be in [0, 1], got '
                f'{self.share!r}.')
        if not (np.isfinite(self.allowance) and self.allowance >= 0):
            raise ValueError(
                f'ProfitCommissionTerms: allowance must be finite and >= 0, got '
                f'{self.allowance!r}.')

    def phi(self, loss_ratio):
        """Profit-commission fraction ``share * (1 - LR - allowance)_+``."""
        lr = np.asarray(loss_ratio, dtype=float)
        return self.share * np.maximum(1.0 - lr - self.allowance, 0.0)


# ======================================================================
# loss-ratio corridor
# ======================================================================
@dataclass(frozen=True)
class CorridorTerms(ContractTerms):
    r"""Loss-ratio corridor: the cedant retains a share of losses in an LR band.

    The cedant keeps ``share`` of the ceded loss whose loss ratio falls in the band
    ``[attachment, attachment + width]`` (in LR points), reducing the cession
    there. DecL ``corridor <share> po <width> xs <attachment>`` (the ``po`` / ``xs``
    layer convention: ``width`` part-of, ``attachment`` excess). Fills the
    **ceded-loss** leg.

    Working in loss-ratio space keeps ``phi`` a pure single-argument map: the ceded
    loss ratio becomes ``LR' = LR - share * clip(LR - attachment, 0, width)`` and
    the analysis layer scales by ceded premium to recover ceded currency
    (``A' = LR' * P``).

    Parameters
    ----------
    share : float
        Cedant's retained share within the corridor, in ``[0, 1]``.
    width : float
        Corridor width in loss-ratio points (the ``po`` part-of); finite, ``> 0``.
    attachment : float
        Corridor attachment loss ratio (the ``xs`` excess); finite, ``>= 0``.

    Notes
    -----
    ``phi(LR) = LR - share * clip(LR - attachment, 0, width)`` -- nondecreasing in
    ``LR`` (a comonotone reduction of the cession), equal to ``LR`` below the band
    and reduced by the constant ``share * width`` above it.

    Examples
    --------
    ``share=0.5, width=0.30, attachment=0.20`` with ceded premium ``P=1000``
    (worked example, reading ``LR = A / P``): ``A=100`` (LR 0.10) -> ``100``;
    ``A=350`` (LR 0.35) -> ``275``; ``A=600`` (LR 0.60) -> ``450``.

    >>> c = CorridorTerms(share=0.5, width=0.30, attachment=0.20)
    >>> c.phi(np.array([0.10, 0.35, 0.60])) * 1000.0
    array([100., 275., 450.])
    """

    target_leg = 'ceded_loss'
    loss_basis = 'ceded_lr'

    share: float
    width: float
    attachment: float

    def __post_init__(self):
        if not (np.isfinite(self.share) and 0.0 <= self.share <= 1.0):
            raise ValueError(
                f'CorridorTerms: share must be in [0, 1], got {self.share!r}.')
        if not (np.isfinite(self.width) and self.width > 0):
            raise ValueError(
                f'CorridorTerms: width must be finite and > 0, got '
                f'{self.width!r}.')
        if not (np.isfinite(self.attachment) and self.attachment >= 0):
            raise ValueError(
                f'CorridorTerms: attachment must be finite and >= 0, got '
                f'{self.attachment!r}.')

    def phi(self, loss_ratio):
        """Retained loss ratio ``LR - share * clip(LR - attachment, 0, width)``."""
        lr = np.asarray(loss_ratio, dtype=float)
        return lr - self.share * np.clip(lr - self.attachment, 0.0, self.width)
