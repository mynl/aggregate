"""Bivariate copulas for coupling per-claim severities in multivariate aggregates.

Provides :class:`Copula`, a small family of two-dimensional copulas built in the
same registry / factory style as :class:`aggregate.spectral.Distortion`: each
concrete kind is a subclass that registers itself by a ``kind`` string, and the
factory call ``Copula('gumbel', 0.4)`` dispatches on the name and returns the
matching subclass instance.

A copula :math:`C(u, v)` is a joint CDF on the unit square with uniform
marginals; by Sklar's theorem it is exactly the dependence structure that can be
glued onto any pair of marginal distributions. Here the marginals are the two
per-claim severities of a :class:`aggregate.multivariate.MultivariateAggregate`,
and the copula sets how the two perils move together within a single event
(e.g. gumbel = both large together, clayton = both small together).

Each kind is parametrised by its **natural** dependence parameter -- the one an
actuary would quote -- and converts internally to the analytic parameter:

================ ====================== ============================
kind             natural parameter      internal
================ ====================== ============================
``independent``  -- (none)              --
``normal``       Pearson ``rho``        ``rho``
``gumbel``       Kendall ``tau``        ``theta = 1 / (1 - tau)``
``clayton``      Kendall ``tau``        ``theta = 2 tau / (1 - tau)``
``fgm``          Spearman ``rho_s``     ``alpha = 3 rho_s``
================ ====================== ============================

Nothing here is re-exported at the top-level package namespace (submodule access
only, per the project layout convention): reach it as
``from aggregate.copula import Copula``.

Notes
-----
The two-parameter ``t`` copula (``rho`` + ``df``) is deferred -- the bivariate-t
CDF needs either a flaky scipy path or a Genz quadrature and is scheduled as a
fast-follow.
"""

import logging

import numpy as np
from scipy.stats import norm, multivariate_normal

logger = logging.getLogger(__name__)

# Clip applied to CDF arguments before an inverse-normal transform so the
# boundary breakpoints u in {0, 1} do not produce +/- inf scores. The copula
# CDF at the boundary is fixed analytically (C(0, v) = 0, C(1, v) = v), so this
# only affects intermediate scratch values that are subsequently masked out.
_PPF_CLIP = 1e-15


class Copula:
    """Base class for bivariate copulas; registry + factory dispatch.

    Each concrete kind is a subclass declared below (:class:`CopulaNormal`,
    :class:`CopulaGumbel`, ...). Subclasses register themselves by setting the
    class attribute ``kind`` (the lookup key) and are auto-collected by
    ``__init_subclass__``.

    The factory call ``Copula('gumbel', 0.4)`` is the primary API: it dispatches
    on the name string and returns an instance of the matching subclass. Direct
    subclass construction (``CopulaGumbel(tau=0.4)``) is also supported.

    Subclass contract
    -----------------
    * ``kind : str`` -- registry key, e.g. ``'gumbel'``.
    * ``param_name : str | None`` -- natural-parameter keyword (``'rho'``,
      ``'tau'``, ``'rho_s'``); ``None`` for the parameter-free ``independent``.
    * ``long_name : str`` -- display label.
    * ``def _build(self): ...`` -- validate :attr:`param` and set the internal
      analytic parameter (``self.theta`` / ``self.rho`` / ``self.alpha``).
    * ``def _C_interior(self, u, v): ...`` -- the copula CDF on the open square
      ``0 < u, v < 1`` (boundaries are handled by the base :meth:`C`).
    * ``def tau(self): ...`` -- Kendall's tau, for reporting.

    Parameters
    ----------
    name : str, optional
        Copula kind, e.g. ``'gumbel'``. When constructing a subclass directly
        ``name`` defaults to the subclass's ``kind``.
    param : float, optional
        Positional natural parameter. May also be passed by its natural name
        (e.g. ``tau=0.4``); passing both raises ``TypeError``.
    display_name : str, optional
        Override label; ``str(c)`` returns this if set.
    **natural : float
        Accept the kind's natural parameter name as a keyword.
    """

    # registry of subclasses keyed by ``kind``; populated by
    # ``__init_subclass__``. Insertion order = declaration order.
    _registry: dict[str, type] = {}

    kind: str = ''
    param_name: str | None = None
    long_name: str = ''

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kind:
            Copula._registry[cls.kind] = cls

    def __new__(cls, name=None, *args, **kwargs):
        """Factory dispatch: ``Copula('gumbel', 0.4)`` -> :class:`CopulaGumbel`.

        When called on a subclass directly, or with ``name=None`` (as happens
        during pickle reconstruction), no dispatch occurs.
        """
        if cls is not Copula or name is None:
            return object.__new__(cls)
        subclass = cls._registry.get(name)
        if subclass is None:
            raise ValueError(
                f"Unknown copula kind {name!r}; "
                f"available: {sorted(cls._registry)}")
        return object.__new__(subclass)

    def __init__(self, name=None, param=None, *, display_name='', **natural):
        if name is None:
            name = type(self).kind
        pn = type(self).param_name
        if pn is not None and pn in natural:
            if param is not None:
                raise TypeError(f'Pass {pn}= or positional param, not both')
            param = natural.pop(pn)
        if natural:
            raise TypeError(
                f'{type(self).__name__}: unexpected keyword arguments '
                f'{list(natural)}')
        self._name = name
        self.param = None if param is None else float(param)
        self.display_name = display_name
        self._build()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build(self):
        """Subclass hook: validate :attr:`param`, set the analytic parameter."""
        return None

    def _C_interior(self, u, v):
        """Copula CDF on the open unit square ``0 < u, v < 1`` (subclass)."""
        raise NotImplementedError(
            f"{type(self).__name__} must override _C_interior()")

    def tau(self):
        """Kendall's tau implied by the copula parameter (subclass)."""
        raise NotImplementedError(
            f"{type(self).__name__} must override tau()")

    # ------------------------------------------------------------------
    # CDF with boundary handling, and the discrete-Sklar rectangle builder
    # ------------------------------------------------------------------

    def C(self, u, v):
        """Copula CDF ``C(u, v)``, vectorised, with exact boundary values.

        Parameters
        ----------
        u, v : array_like
            Values in ``[0, 1]`` (broadcast together).

        Returns
        -------
        ndarray or float
            ``C(u, v)``. The boundary conditions ``C(0, v) = C(u, 0) = 0``,
            ``C(1, v) = v`` and ``C(u, 1) = u`` are enforced exactly; the
            subclass :meth:`_C_interior` supplies the open-square values.

        Notes
        -----
        ``np.where`` evaluates the interior formula everywhere (including the
        boundary, where it may be ``nan``/``inf``); those values are then
        overwritten by the analytic boundary conditions, so a blanket
        ``errstate(all='ignore')`` is appropriate.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)
        with np.errstate(all='ignore'):
            interior = np.asarray(self._C_interior(u, v), dtype=float)
        out = np.where((u > 0) & (v > 0), interior, 0.0)
        # boundary conditions (applied after the interior fill)
        out = np.where(u >= 1.0, v, out)
        out = np.where(v >= 1.0, u, out)
        out = np.where((u <= 0.0) | (v <= 0.0), 0.0, out)
        return out[()] if out.ndim == 0 else out

    def rectangle_pmf(self, G1, G2):
        """Discrete-Sklar joint pmf from two marginal CDFs.

        Given marginal per-claim CDF breakpoints ``G1`` (length ``n1``) and
        ``G2`` (length ``n2``) -- the cumulative sums of the marginal severity
        masses, each ending at ~1 and possibly with a jump at the zero bucket --
        the joint mass in cell ``(i, j)`` is the copula rectangle probability

        .. math::

            S[i, j] = C(G_1[i], G_2[j]) - C(G_1[i-1], G_2[j])
                      - C(G_1[i], G_2[j-1]) + C(G_1[i-1], G_2[j-1])

        with ``G[-1] := 0``.

        Parameters
        ----------
        G1, G2 : array_like
            Marginal CDF breakpoints (cumulative severity masses).

        Returns
        -------
        ndarray
            Joint pmf, shape ``(n1, n2)``. Row sums equal ``diff([0, *G1])``
            and column sums equal ``diff([0, *G2])`` -- i.e. the marginals are
            reproduced exactly, for any copula and any (including atomic)
            marginals.

        Notes
        -----
        Evaluating ``C`` once on the ``(n1+1) x (n2+1)`` outer grid of
        breakpoints (with a leading 0 prepended to each axis) and taking the
        second mixed difference is ``O(n1 n2)`` -- cheap for the closed-form
        copulas, ~seconds for the Gaussian CDF at a few hundred per axis.
        """
        G1 = np.concatenate([[0.0], np.asarray(G1, dtype=float)])
        G2 = np.concatenate([[0.0], np.asarray(G2, dtype=float)])
        cmat = self.C(G1[:, None], G2[None, :])
        return (cmat[1:, 1:] - cmat[:-1, 1:]
                - cmat[1:, :-1] + cmat[:-1, :-1])

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def __repr__(self):
        if self.param is None:
            return f'Copula({self._name!r})'
        return (f'Copula({self._name!r}, {type(self).param_name}={self.param:g}'
                f' -> tau={self.tau():.4f})')

    def __str__(self):
        if self.display_name:
            return self.display_name
        if self.param is None:
            return self._name
        return f'{self._name}({type(self).param_name}={self.param:g})'


# ---------------------------------------------------------------------------
# Concrete copulas
# ---------------------------------------------------------------------------

class CopulaIndependent(Copula):
    """The independence copula ``C(u, v) = u v`` (parameter-free)."""

    kind = 'independent'
    param_name = None
    long_name = 'Independence'

    def _build(self):
        self.param = None

    def _C_interior(self, u, v):
        return u * v

    def tau(self):
        """Kendall's tau (zero under independence)."""
        return 0.0


class CopulaNormal(Copula):
    """Gaussian copula, natural parameter Pearson ``rho`` in ``(-1, 1)``.

    ``C(u, v) = Phi_rho(Phi^{-1} u, Phi^{-1} v)`` where ``Phi_rho`` is the
    standard bivariate normal CDF with correlation ``rho`` and ``Phi^{-1}`` the
    standard normal quantile. Full dependence range, **no** tail dependence.
    """

    kind = 'normal'
    param_name = 'rho'
    long_name = 'Gaussian'

    def _build(self):
        if self.param is None or not (-1.0 < self.param < 1.0):
            raise ValueError(
                f'normal copula: rho must be in (-1, 1), got {self.param!r}')
        self.rho = self.param
        self._cov = np.array([[1.0, self.rho], [self.rho, 1.0]])

    @classmethod
    def from_tau(cls, tau):
        """Construct from Kendall's tau via ``rho = sin(pi tau / 2)``."""
        return cls(rho=float(np.sin(np.pi * tau / 2.0)))

    def _C_interior(self, u, v):
        x = norm.ppf(np.clip(u, _PPF_CLIP, 1.0 - _PPF_CLIP))
        y = norm.ppf(np.clip(v, _PPF_CLIP, 1.0 - _PPF_CLIP))
        pts = np.stack([np.asarray(x).ravel(), np.asarray(y).ravel()], axis=-1)
        cdf = multivariate_normal.cdf(pts, mean=[0.0, 0.0], cov=self._cov)
        return np.asarray(cdf).reshape(np.asarray(x).shape)

    def tau(self):
        """Kendall's tau ``= (2 / pi) arcsin(rho)``."""
        return float(2.0 / np.pi * np.arcsin(self.rho))


class CopulaGumbel(Copula):
    """Gumbel copula, natural parameter Kendall ``tau`` in ``[0, 1)``.

    ``C(u, v) = exp(-((-ln u)^theta + (-ln v)^theta)^{1/theta})`` with
    ``theta = 1 / (1 - tau) >= 1``. Upper-tail dependence (both perils large
    together); ``tau = 0`` is independence.
    """

    kind = 'gumbel'
    param_name = 'tau'
    long_name = 'Gumbel'

    def _build(self):
        if self.param is None or not (0.0 <= self.param < 1.0):
            raise ValueError(
                f'gumbel copula: tau must be in [0, 1), got {self.param!r}')
        self.theta = 1.0 / (1.0 - self.param)

    def _C_interior(self, u, v):
        lu = -np.log(u)
        lv = -np.log(v)
        return np.exp(-(lu ** self.theta + lv ** self.theta) ** (1.0 / self.theta))

    def tau(self):
        """Kendall's tau ``= 1 - 1/theta`` (the natural parameter)."""
        return float(1.0 - 1.0 / self.theta)


class CopulaClayton(Copula):
    """Clayton copula, natural parameter Kendall ``tau`` in ``[0, 1)``.

    ``C(u, v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}`` with
    ``theta = 2 tau / (1 - tau) > 0``. Lower-tail dependence (both perils small
    together); ``tau = 0`` is independence.
    """

    kind = 'clayton'
    param_name = 'tau'
    long_name = 'Clayton'

    def _build(self):
        if self.param is None or not (0.0 <= self.param < 1.0):
            raise ValueError(
                f'clayton copula: tau must be in [0, 1), got {self.param!r}')
        self.theta = 2.0 * self.param / (1.0 - self.param)

    def _C_interior(self, u, v):
        if self.theta <= 0.0:        # tau == 0 -> independence
            return u * v
        return (u ** (-self.theta) + v ** (-self.theta) - 1.0) ** (-1.0 / self.theta)

    def tau(self):
        """Kendall's tau ``= theta / (theta + 2)`` (the natural parameter)."""
        return float(self.theta / (self.theta + 2.0))


class CopulaFGM(Copula):
    """Farlie-Gumbel-Morgenstern copula, natural parameter Spearman ``rho_s``.

    ``C(u, v) = u v (1 + alpha (1 - u)(1 - v))`` with ``alpha = 3 rho_s`` in
    ``[-1, 1]`` (so ``rho_s`` in ``[-1/3, 1/3]``). Captures only weak
    dependence, no tail dependence; useful as a light perturbation of
    independence with either sign.
    """

    kind = 'fgm'
    param_name = 'rho_s'
    long_name = 'Farlie-Gumbel-Morgenstern'

    def _build(self):
        if self.param is None or not (-1.0 / 3.0 - 1e-12 <= self.param
                                      <= 1.0 / 3.0 + 1e-12):
            raise ValueError(
                f'fgm copula: rho_s must be in [-1/3, 1/3], got {self.param!r}')
        self.alpha = 3.0 * self.param

    def _C_interior(self, u, v):
        return u * v * (1.0 + self.alpha * (1.0 - u) * (1.0 - v))

    def tau(self):
        """Kendall's tau ``= 2 alpha / 9``."""
        return float(2.0 * self.alpha / 9.0)
