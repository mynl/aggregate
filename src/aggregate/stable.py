"""Alpha-stable laws: parameters, their algebra and closed-form facts.

:class:`Stable` holds the four parameters of an alpha-stable law in Nolan's S1
parameterization, converts to and from S0, adds, scales and negates, and
reports the characteristic function, the mean, the probability above the
location and the power tail constants in closed form. Distribution values
(``pdf``, ``cdf``, ``ppf`` and the rest) come from ``scipy.stats.levy_stable``,
which is reliable in the body and unreliable in the far tail; see the class
Notes.

The answer to a bare ``stable`` DecL statement, ``build('stable 1.5 1 2 0')``,
is one of these, and so is a named law, ``stable x0 1.5 0``, stored in the recipe
base and referenced as ``stable.x0``. Submodule access only: ``from
aggregate.stable import Stable``, as for ``aggregate.tweedie.Tweedie``. A leaf
module, importing only numpy, scipy and ``_labeled`` (which imports nothing of
this package), so the parser can import it lazily.

Moved from PMIR's ``pmir_legacy/stable.py`` (March 2025) without its claim
count emergence methods, which stay in PMIR. See
``dev/done/plan-stable-statement.md``.
"""

import numbers

import numpy as np
import scipy.stats as ss
from scipy.special import gamma as gamma_function

from ._labeled import LabeledMixin

__all__ = ['Stable']


def _fmt_full(x):
    """A number as DecL text that parses back to the same float."""
    x = float(x) + 0.0
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def _fmt_short(x):
    """A number to six significant figures, never ``-0``."""
    return f'{float(x) + 0.0:.6g}'


def _is_real(x):
    """A real scalar, excluding ``bool`` (which is an ``int``)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


class Stable(LabeledMixin):
    r"""The parameters of an alpha-stable law, in Nolan's S1 and S0 forms.

    Parameters
    ----------
    alpha : float
        The index of stability, ``0 < alpha <= 2``.
    beta : float
        The skewness, ``-1 <= beta <= 1``. It has no effect at ``alpha = 2``,
        where it is set to 0.
    scale : float, default 1
        The scale ``gamma > 0``.
    loc : float, default 0
        The location in the parameterization named by ``parameterization``.
    parameterization : {'S1', 'S0'}, default 'S1'
        How ``loc`` is read, case-insensitive. The object always stores the S1
        location as :attr:`loc`; :attr:`loc0` is the S0 location.
    name : str, optional
        The handle of a law declared with ``stable NAME ...``. Arithmetic returns
        an anonymous law.
    label : str, optional
        The display label, the DecL ``as`` clause.

    Attributes
    ----------
    alpha, beta, scale, loc : float
        The S1 parameters.
    name : str or None
        The handle, ``None`` for an anonymous law.
    program : str
        The DecL statement, set when the recipe base builds a declaration, else
        empty.

    Notes
    -----
    **Notation.** Three spellings of the same four numbers are in use:

    ================  ==========================================  ==========
    Nolan             PMIR ``pmir_legacy.stable``                 scipy
    ================  ==========================================  ==========
    :math:`\alpha`    ``a``                                       ``alpha``
    :math:`\beta`     ``b``                                       ``beta``
    :math:`\gamma`    ``c``                                       ``scale``
    :math:`\delta`    ``d``                                       ``loc``
    ================  ==========================================  ==========

    **Parameterizations** [Nolan 2020, Def. 1.3 to 1.7]. In S1, for
    :math:`\alpha \ne 1`,
    :math:`\log \mathsf{P} e^{itX} = -\gamma^\alpha|t|^\alpha(1 - i\beta\,\mathrm{sign}(t)\tan(\pi\alpha/2)) + i\delta_1 t`,
    and at :math:`\alpha = 1` the bracket is
    :math:`1 + i\beta(2/\pi)\,\mathrm{sign}(t)\log|t|` with :math:`\gamma|t|` in
    front. The S0 location is :math:`\delta_0 = \delta_1 + \beta\gamma\tan(\pi\alpha/2)`,
    or :math:`\delta_1 + (2/\pi)\beta\gamma\log\gamma` at :math:`\alpha = 1`.
    S1 is scipy's default and the form in which sums and scaling are clean;
    S0 is continuous in :math:`\alpha` and Nolan's choice for numerical work.
    At :math:`\alpha = 2` the law is normal with variance :math:`2\gamma^2`.

    **Algebra, in S1** [Nolan 2020, Prop. 1.3 and 1.4]. Independent laws with a
    common :math:`\alpha` add: :math:`\gamma^\alpha` adds, :math:`\beta` is the
    :math:`\gamma^\alpha` weighted average, and locations add. For a real
    :math:`a \ne 0`, :math:`aX` has skewness :math:`\mathrm{sign}(a)\beta`,
    scale :math:`|a|\gamma` and location :math:`a\delta_1`, less
    :math:`(2/\pi)\beta\gamma a\log|a|` when :math:`\alpha = 1`. Adding a
    number shifts the location. ``sum()`` works, because ``0 + X`` is ``X``.

    **Independent copies.** These operations act on laws, not on random
    variables, and each operand is an independent copy. Hence ``X + X`` is not
    ``2 * X``: at :math:`\alpha = 3/2` the first has scale
    :math:`2^{2/3} = 1.5874` and the second has scale 2, and that difference is
    the stability property itself. Likewise ``X - X`` is a symmetric stable law
    with scale 1.5874, not zero. ``t @ X`` is the sum of ``t`` independent copies,
    with characteristic function :math:`\varphi^t`, so scale
    :math:`t^{1/\alpha}\gamma` and location :math:`t\delta_1`. This is the law at
    time ``t`` of the Levy process with :math:`X` at time 1, and ``t`` need not be
    an integer. At :math:`\alpha = 1` it has no logarithm term, which only a
    change of scale brings.

    **Facts.** The mean is :math:`\delta_1` for :math:`\alpha > 1` and does not
    exist otherwise. For :math:`\alpha \ne 1`, :math:`X - \delta_1` is strictly
    stable and :math:`\Pr(X > \delta_1) = 1/2 + \arctan(\beta\tan(\pi\alpha/2))/(\pi\alpha)`,
    which is :math:`1 - 1/\alpha` when :math:`\beta = 1` and
    :math:`1 < \alpha < 2`. For :math:`\alpha < 2` the tails are
    :math:`\Pr(X > x) \sim (1+\beta)C_\alpha\gamma^\alpha x^{-\alpha}` and
    :math:`\Pr(X < -x) \sim (1-\beta)C_\alpha\gamma^\alpha x^{-\alpha}`, with
    :math:`C_\alpha = \Gamma(\alpha)\sin(\pi\alpha/2)/\pi` [Nolan 2020, Thm. 1.2].

    **scipy in the far tail.** ``pdf``, ``cdf``, ``sf``, ``ppf``, ``isf`` and
    ``rvs`` delegate to ``scipy.stats.levy_stable``, which is accurate in the
    body and unreliable far out. Measured with scipy 1.17.1 at
    :math:`\alpha = 1.5`, :math:`\beta = 1`, unit scale: ``sf(659)`` is
    :math:`2.358\times10^{-5}` and agrees with the tail asymptotic, yet
    ``ppf(1 - 1e-6)`` returns 659 rather than about 5430, and ``sf(5430)``
    returns 0 against an asymptotic :math:`9.97\times10^{-7}`. ``pdf`` at 2,000
    points takes about 0.7 s. Use :attr:`tail_constant` in the far tail.
    Accurate distribution values by inverting :meth:`chf` are
    ``[Characteristic-Function-Engine]``, a parked plan.

    ``fz`` is built in whichever parameterization the scipy class attribute
    ``ss.levy_stable.parameterization`` names when it is first used, and the
    attribute is never assigned here, because it is shared by every caller in
    the process.

    Examples
    --------
    >>> from aggregate.stable import Stable
    >>> print(Stable(1.5, 1, 2, 0))
    stable 1.5 1 2 0 = stable 1.5 1 2 -2 s0, mean 0, Pr(X>mean) 0.333333, Pr(X>x) ~ 1.12838 x^-1.5
    >>> X = Stable(1.5, 1, 2) + Stable(1.5, -1, 2)
    >>> X.beta, round(X.scale, 6)
    (0.0, 3.174802)
    """

    #: The DecL statement; the recipe base sets this on an object it builds.
    program = ''

    def __init__(self, alpha, beta, scale=1.0, loc=0.0, parameterization='S1',
                 name=None, label=None):
        param = str(parameterization).upper()
        if param not in ('S0', 'S1'):
            raise ValueError(
                f"Stable: parameterization must be 'S0' or 'S1', got "
                f"{parameterization!r}")
        for what, x in (('alpha', alpha), ('beta', beta), ('scale', scale),
                        ('loc', loc)):
            if not _is_real(x):
                raise ValueError(f'Stable: {what} must be a real number, got {x!r}')
        alpha, beta, scale, loc = float(alpha), float(beta), float(scale), float(loc)
        if not 0 < alpha <= 2:
            raise ValueError(f'Stable: alpha must be in (0, 2], got {alpha}')
        if not -1 <= beta <= 1:
            raise ValueError(f'Stable: beta must be in [-1, 1], got {beta}')
        if not 0 < scale < np.inf:
            raise ValueError(f'Stable: scale must be positive and finite, got {scale}')
        if not np.isfinite(loc):
            raise ValueError(f'Stable: loc must be finite, got {loc}')
        if alpha == 2:
            # normal: the skewness has no effect, so normalize it away
            beta = 0.0
        if param == 'S0':
            loc = float(self.s0_to_s1(alpha, beta, scale, loc))
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.loc = loc
        self.name = name
        self._init_labels(label=label)
        self._fz = None

    @classmethod
    def from_spec(cls, spec):
        """The law a stored ``stable`` spec describes.

        Parameters
        ----------
        spec : dict
            Holds ``alpha``, ``beta``, ``scale`` and ``loc``, the location in
            S1, and optionally ``name`` and ``label``. The trailer keys are
            ignored.

        Returns
        -------
        Stable
        """
        return cls(spec['alpha'], spec['beta'], spec['scale'], spec['loc'],
                   name=spec.get('name'), label=spec.get('label'))

    def _label_default(self):
        """The short spelling, since an anonymous law has no handle to fall back on."""
        return ' '.join(['stable', *(_fmt_short(x) for x in
                                     (self.alpha, self.beta, self.scale, self.loc))])

    # ----- parameterizations -----------------------------------------
    @staticmethod
    def _location_shift(alpha, beta, scale):
        """``loc0 - loc1``: what the S0 location adds to the S1 location."""
        if alpha == 1:
            return 2 / np.pi * beta * scale * np.log(scale)
        return beta * scale * np.tan(np.pi * alpha / 2)

    @staticmethod
    def s0_to_s1(alpha, beta, scale, loc0):
        """The S1 location of the law whose S0 location is ``loc0``.

        Parameters
        ----------
        alpha, beta, scale : float
            The parameters the two forms share.
        loc0 : float
            The S0 location.

        Returns
        -------
        float
        """
        return loc0 - Stable._location_shift(alpha, beta, scale)

    @staticmethod
    def s1_to_s0(alpha, beta, scale, loc1):
        """The S0 location of the law whose S1 location is ``loc1``.

        Parameters
        ----------
        alpha, beta, scale : float
            The parameters the two forms share.
        loc1 : float
            The S1 location.

        Returns
        -------
        float
        """
        return loc1 + Stable._location_shift(alpha, beta, scale)

    @property
    def loc0(self):
        """The S0 location."""
        return self.s1_to_s0(self.alpha, self.beta, self.scale, self.loc)

    # ----- closed forms ----------------------------------------------
    def chf(self, t):
        r"""The characteristic function :math:`\mathsf{P}e^{itX}`.

        Parameters
        ----------
        t : float or array_like

        Returns
        -------
        complex ndarray

        Notes
        -----
        The S1 closed form in the class Notes. At :math:`\alpha = 1` the
        :math:`|t|\log|t|` term is 0 at :math:`t = 0`; the logarithm is taken
        only where :math:`t \ne 0`, so no warning is raised there.
        """
        t = np.asarray(t, dtype=float)
        a, b, g, d = self.alpha, self.beta, self.scale, self.loc
        abs_t = np.abs(t)
        if a == 1:
            log_t = np.log(abs_t, where=abs_t > 0, out=np.zeros_like(abs_t))
            psi = -g * abs_t * (1 + 1j * b * (2 / np.pi) * np.sign(t) * log_t)
        else:
            psi = -(g * abs_t) ** a * (1 - 1j * b * np.sign(t) * np.tan(np.pi * a / 2))
        return np.exp(psi + 1j * d * t)

    @property
    def mean(self):
        """The mean: ``loc`` for ``alpha > 1``, else infinite or undefined.

        For ``alpha <= 1`` the mean is ``inf`` when ``beta = 1`` (the left tail
        is light and the right tail heavy), ``-inf`` when ``beta = -1``, and
        ``nan`` otherwise.
        """
        if self.alpha > 1:
            return self.loc
        if self.beta == 1:
            return np.inf
        if self.beta == -1:
            return -np.inf
        return np.nan

    @property
    def prob_above_loc(self):
        """``Pr(X > loc)``, the S1 location, in closed form.

        ``nan`` at ``alpha = 1`` unless ``beta = 0``, where it is 1/2: there
        ``X - loc`` is not strictly stable and the closed form does not apply.
        """
        a, b = self.alpha, self.beta
        if a == 2:
            return 0.5
        if a == 1:
            return 0.5 if b == 0 else np.nan
        return 0.5 + np.arctan(b * np.tan(np.pi * a / 2)) / (np.pi * a)

    def _c_alpha_gamma(self):
        """``C_alpha * scale ** alpha``, the tail constant before the skew factor."""
        a = self.alpha
        if a == 2:
            return 0.0
        return gamma_function(a) * np.sin(np.pi * a / 2) / np.pi * self.scale ** a

    @property
    def tail_constant(self):
        """``A`` in ``Pr(X > x) ~ A x ** -alpha``; 0 for the normal."""
        return (1 + self.beta) * self._c_alpha_gamma()

    @property
    def left_tail_constant(self):
        """``B`` in ``Pr(X < -x) ~ B x ** -alpha``; 0 for the normal."""
        return (1 - self.beta) * self._c_alpha_gamma()

    # ----- scipy -----------------------------------------------------
    @property
    def fz(self):
        """The frozen scipy distribution, built on first use.

        ``scipy.stats.levy_stable`` in the parameterization its class attribute
        names when this is first read, with the matching location; a frozen
        object keeps that parameterization. ``scipy.stats.norm`` with
        standard deviation ``sqrt(2) * scale`` at ``alpha = 2``.
        """
        if self._fz is None:
            if self.alpha == 2:
                self._fz = ss.norm(loc=self.loc, scale=np.sqrt(2) * self.scale)
            else:
                fz = ss.levy_stable(self.alpha, self.beta, loc=self.loc,
                                    scale=self.scale)
                if fz.dist.parameterization == 'S0':
                    fz = ss.levy_stable(self.alpha, self.beta, loc=self.loc0,
                                        scale=self.scale)
                self._fz = fz
        return self._fz

    def pdf(self, x):
        """Density, from scipy; see the class Notes on the far tail."""
        return self.fz.pdf(x)

    def cdf(self, x):
        """Distribution function, from scipy; see the class Notes on the far tail."""
        return self.fz.cdf(x)

    def sf(self, x):
        """Survival function, from scipy; see the class Notes on the far tail."""
        return self.fz.sf(x)

    def ppf(self, p):
        """Quantile function, from scipy; see the class Notes on the far tail."""
        return self.fz.ppf(p)

    def isf(self, p):
        """Inverse survival function, from scipy; see the class Notes on the far tail."""
        return self.fz.isf(p)

    def rvs(self, size=None, random_state=None):
        """Random variates, from scipy."""
        return self.fz.rvs(size=size, random_state=random_state)

    def plot(self, kind='density', n=101, p=1e-3, ax=None, **kwargs):
        """Plot the density or the quantile function, from scipy.

        Parameters
        ----------
        kind : {'density', 'quantile'}
            The density between the ``p`` and ``1 - p`` quantiles, or the
            quantile function on ``[p/2, 1 - p/2]``.
        n : int
            Number of points.
        p : float
            Probability left out at each end.
        ax : matplotlib Axes, optional
            Created, 3 by 3 inches, when omitted.
        **kwargs
            Passed to ``ax.set``.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
        if kind == 'density':
            lo, hi = self.ppf([p, 1 - p])
            xs = np.linspace(lo, hi, n)
            ax.plot(xs, self.pdf(xs), label=repr(self))
            ax.set(title='Density', **kwargs)
        elif kind == 'quantile':
            ps = np.linspace(p / 2, 1 - p / 2, n)
            ax.plot(ps, self.ppf(ps), label=repr(self))
            ax.set(title='Quantile function', **kwargs)
        else:
            raise ValueError(f"Stable.plot: kind must be 'density' or 'quantile', got {kind!r}")
        return ax

    # ----- algebra ---------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Stable):
            if other.alpha != self.alpha:
                raise ValueError(
                    f'Stable: only laws with a common alpha add to a stable law, '
                    f'got {self.alpha} and {other.alpha}')
            a = self.alpha
            w1, w2 = self.scale ** a, other.scale ** a
            beta = float(np.clip((self.beta * w1 + other.beta * w2) / (w1 + w2), -1, 1))
            return Stable(a, beta, (w1 + w2) ** (1 / a), self.loc + other.loc)
        if _is_real(other):
            return Stable(self.alpha, self.beta, self.scale, self.loc + other)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        if isinstance(other, Stable) or _is_real(other):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        if _is_real(other):
            return (-self) + other
        return NotImplemented

    def __mul__(self, other):
        if not _is_real(other):
            return NotImplemented
        a = float(other)
        if a == 0:
            raise ValueError('Stable: scaling by zero gives a point mass, not a stable law')
        loc = a * self.loc
        if self.alpha == 1:
            loc -= 2 / np.pi * self.beta * self.scale * a * np.log(abs(a))
        return Stable(self.alpha, np.sign(a) * self.beta, abs(a) * self.scale, loc)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not _is_real(other):
            return NotImplemented
        if other == 0:
            raise ValueError('Stable: division by zero does not give a stable law')
        return self * (1 / other)

    def __rmatmul__(self, other):
        r"""``t @ X``: the sum of ``t`` independent copies of this law.

        Parameters
        ----------
        other : float
            The number of copies ``t``, positive and not necessarily an integer.

        Returns
        -------
        Stable
            The same ``alpha`` and ``beta``, scale ``t ** (1 / alpha) * scale`` and
            location ``t * loc``. Anonymous, like every operator's result.

        Notes
        -----
        The characteristic function is :math:`\varphi^t`, by infinite divisibility.
        Raising the S1 form to the power :math:`t` multiplies the exponent, so
        :math:`\gamma^\alpha` and :math:`\delta_1` scale by :math:`t` and the skewness
        is unchanged, at every :math:`\alpha`. The logarithm term at
        :math:`\alpha = 1` in :meth:`__mul__` comes from a change of scale and does
        not arise here. The ``@`` is the one on an ``agg``, where ``2 @ agg.X``
        doubles the claim count.
        """
        if not _is_real(other):
            return NotImplemented
        t = float(other)
        if not 0 < t < np.inf:
            raise ValueError(
                f'Stable: the number of independent copies in t @ X must be positive '
                f'and finite, got {other}')
        return Stable(self.alpha, self.beta, t ** (1 / self.alpha) * self.scale, t * self.loc)

    # ----- the generalized central limit theorem ----------------------
    @classmethod
    def from_compound_pareto(cls, frequency, alpha, theta, sev_mean):
        r"""The stable limit of a Poisson compound with a Pareto tail.

        Parameters
        ----------
        frequency : float
            The expected Poisson claim count :math:`\lambda`.
        alpha : float
            The severity tail index, ``1 < alpha < 2``.
        theta : float
            The severity tail scale: :math:`\Pr(X > x) = \theta^\alpha x^{-\alpha}(1 + o(1))`.
        sev_mean : float
            The severity mean.

        Returns
        -------
        Stable
            :math:`S(\alpha, 1, \gamma, \lambda\,\mathsf{P}X; 1)`.

        Notes
        -----
        The compound has Levy measure :math:`\lambda F(dx)`, with tail
        :math:`\lambda\theta^\alpha x^{-\alpha}`, and for large :math:`\lambda`
        it is approximately stable with
        :math:`\gamma^\alpha = \lambda\theta^\alpha\Gamma(1-\alpha)\cos(\pi\alpha/2)`
        [Feller 1971, ch. XVII]. Both factors are negative on (1, 2), so the
        product is positive. By the reflection formula the right tail constant
        :math:`2C_\alpha\gamma^\alpha` equals :math:`\lambda\theta^\alpha`, so the
        approximation has exactly the compound's mean and right tail; its left
        tail decays faster than any power. For ``alpha <= 1`` the centering is
        different and the mean does not exist, so it is refused.
        """
        if not 1 < alpha < 2:
            raise ValueError(
                f'stable: the limit theorem form needs a severity tail index '
                f'strictly between 1 and 2, got {alpha}')
        for what, x in (('claim count', frequency), ('severity scale', theta)):
            if not x > 0:
                raise ValueError(f'stable: the {what} must be positive, got {x}')
        g = (frequency * theta ** alpha * gamma_function(1 - alpha)
             * np.cos(np.pi * alpha / 2)) ** (1 / alpha)
        return cls(alpha, 1.0, g, frequency * sev_mean)

    # ----- text ------------------------------------------------------
    def to_decl(self):
        """The ``stable`` statement for this law, in S1, at full precision.

        The statement of a named law includes the name and the label, so it
        declares the law again.
        """
        from .decl_writer import _render_label
        head = 'stable'
        if self.name:
            head = f'stable {self.name}{_render_label(self._label)}'
        return (f'{head} {_fmt_full(self.alpha)} {_fmt_full(self.beta)} '
                f'{_fmt_full(self.scale)} {_fmt_full(self.loc)}')

    def __repr__(self):
        extra = ''
        if self.name is not None:
            extra += f', name={self.name!r}'
        if self._label is not None:
            extra += f', label={self._label!r}'
        return (f'Stable(alpha={_fmt_full(self.alpha)}, beta={_fmt_full(self.beta)}, '
                f'scale={_fmt_full(self.scale)}, loc={_fmt_full(self.loc)}{extra})')

    def __str__(self):
        g = _fmt_short

        def g_loc(x):
            # A location is the difference of two numbers of the size of the
            # scale (tan(3 pi / 4) is -1.0000000000000002), so a residue far
            # below the scale is shown as the zero it stands for.
            return g(0.0 if abs(x) <= 1e-12 * self.scale else x)

        head = f'stable {g(self.alpha)} {g(self.beta)} {g(self.scale)}'
        lead = f'{self._title_name}: ' if self.name else ''
        parts = [f'{lead}{head} {g_loc(self.loc)} = {head} {g_loc(self.loc0)} s0']
        mean = self.mean
        if np.isfinite(mean):
            parts.append(f'mean {g_loc(mean)}')
            above = 'mean'
        else:
            parts.append('mean infinite' if mean == np.inf else
                         'mean negatively infinite' if mean == -np.inf else
                         'mean undefined')
            above = 'loc'
        if np.isfinite(self.prob_above_loc):
            parts.append(f'Pr(X>{above}) {g(self.prob_above_loc)}')
        if self.alpha == 2:
            parts.append(f'normal, variance {g(2 * self.scale ** 2)}')
        else:
            power = f'x^-{g(self.alpha)}'
            if self.tail_constant > 0:
                parts.append(f'Pr(X>x) ~ {g(self.tail_constant)} {power}')
            if self.left_tail_constant > 0:
                parts.append(f'Pr(X<-x) ~ {g(self.left_tail_constant)} {power}')
        return ', '.join(parts)
