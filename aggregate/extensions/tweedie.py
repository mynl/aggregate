from enum import Enum
try:
    from jax import grad     # EXTREMELY idle...
    import jax.numpy as jnp
except:
    pass
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats as ss
from typing import Optional, Tuple  # , Union, Any  # Import necessary types
from pathlib import Path

from IPython.display import display
# from collections import namedtuple
# from numpy import ndarray, dtype

from ..distributions import Aggregate  # noqa
from ..utilities import MomentWrangler
from ..underwriter import build
from .ft import FourierTools

# logging
logger = logging.getLogger(__name__)


class Mode(Enum):
    """Mode of the Tweedie class."""
    REPRODUCTIVE = 1
    ADDITIVE = 2


class Tweedie(object):
    """
    ``Tweedie`` is a class for working with the Tweedie class of
    exponential dispersion models, those with variance function

        V(μ) = dispersion x μ^p,

    where p is the power parameter and μ the mean. The Tweedie class
    of distributions includes:

    * 1 < p < 2: Tweedie distributions, a Poisson-gamma compound distributions
    * p = 1: Poisson distributions
    * p = 0: normal distributions
    * p < 0: spectrally positive stable distributions with index 1 < alpha < 2
    * p = infinity: spectrally positive Cauchy distribution with index alpha = 1
    * 2 < p < infinity: positive spectrally positive stable distributions with
      index 0 < alpha < 1
    * p = 3 (special case): inverse Gaussian and Levy(1/2) distributions
    * p = 2: gamma distributions

    The range 0 < p < 1 is impossible and does not correspond to an exponential
    dispersion model.

    The ``Tweedie`` class provides

    * Translation between the reproductive and additive parameterizations. The
      former is used to model averages like pure premiums that combined in weighted
      sums and the latter models totals (sums) that are simply additive.

    * Create frozen ``scipy.stats``-like distributions, either explicitly
      as a ``scipy.stats`` object from the normal, levy_stable, levy, inverse
      Gaussian, or gamma distribution, or as an ``Aggregate`` object otherwise.
    * For Tweedie distributions it translates parameters into Poisson frequency and
      gamma severity, the latter as mean and CV or shape and rate.
    * Create a ``aggregate.extensions.ft.FourierTools`` object for Fourier analysis
      that provides an approximation to the distribution for those without a
      closed form expression (extreme and positive extreme stable, titled Cauchy,
      Tweedie).
    * The generating distribution ``c``.
    * The cumulant function, ``kappa``.
    * The tilted cumulant function, ``K``
    * The characteristic function, ``chf``.
    * The Levy measure density, ``levy_density``, see [3]

    Methodology
    -----------

    The class is initialized with either the reproductive parameters (p, mean, dispersion)
    or the additive parameters (p, theta, index). The class then computes the missing
    parameters. The class creates a frozen distribution object on demand for the
    additive or reproductive distribution per those provided at initialization. Methods
    are available to create the dual. Methods are also available to translate between
    additive and reproductive parameter for the same distribution (not the dual).

    Notation
    =========

    The canonical parameter is always called theta, p is always p. The other parameters
    are named (mean, dispersion, index) or (frequency, shape, rate) or (sev_m, sev_cv).
    The dispersion is usually sigma^2. The index is lambda or phi (but the frequency is
    sometimes lambda).

    Notation follows Jorgensen [1] and [2]. My blog post on Tweedie uses -alpha for the
    shape.

    References
    =============
    [1] Jørgensen, Bent. "Exponential dispersion models." Journal of the Royal Statistical Society
        Series B: Statistical Methodology 49.2 (1987): 127-145.
    [2] Jorgensen, B. (1997). The theory of dispersion models. Chapman and Hall/CRC.
    [3] Jørgensen, Bent, and José Raúl Martínez. "THE LÉVY—KHINCHINE REPRESENTATION OF THE TWEEDIE
        CLASS." Brazilian Journal of Probability and Statistics (1996): 225-233.
    """

    _renamer = {'mean': '$\\mu$', 'p': '$p$', 'dispersion': '$\\phi$', 'theta': '$\\theta$', 'index': '$\\phi$',
                'frequency': '$\\lambda$', 'shape': '$\\alpha$', 'rate': '$\\beta$', 'sev_m': '$\\mu_X$',
                'sev_cv': '$\\nu_X$', 'cv': '$\\nu$', 'p0': '$\\Pr(X=0)$'}

    def __init__(self, p: float, *, mean: Optional[float] = None, dispersion: Optional[float] = 1,
                 theta: Optional[float] = None, index: Optional[float] = 1):
        """
        Initialize object from either reproductive p, mean (mu), dispersion (sigma^2) or additive
        p, theta (canonical), index (phi or lambda) parameters. All parameters except p, which is
        required must be given by name. See also the static method ``Tweedie.from_po_gamma`` to
        create a Tweedie distribution (1 < p < 2) from the claim count and severity parameters.

        Notice that alpha - 1 = -1 / (p - 1) (p. 131 after eq 4.10).

        See Also
        ---------

        * ``Tweedie.dual``: create the dual object.
        * ``Tweedie.from_p_mean_cv``: create a Tweedie object from p, mean, and cv.
        * ``Tweedie.from_po_gamma``: create a Tweedie object from the Poisson-gamma parameters.
        * ``Tweedie.from_p_mean_cv``: create a Tweedie object from p, mean, and cv.


        """
        if 0 < p < 1:
            raise ValueError(f'p cannot be in (0, 1), got {p}')

        alpha = self.p_to_alpha(p)

        if mean is not None and dispersion is not None:
            # reproductive
            self.mode = Mode.REPRODUCTIVE
            if dispersion <= 0:
                raise ValueError('Dispersion must be strictly positive')
            # _, theta, index = self.reproductive_to_additive(p, mean, dispersion)
            index = 1. / dispersion
            # Jorg [2], p 131 equation after 4.12
            if p == 1:
                theta = np.log(mean)
            else:
                # using (alpha - 1)(p - 1) = -1
                theta = (alpha - 1) * mean ** (1 - p)
        elif theta is not None and index is not None:
            # additive
            self.mode = Mode.ADDITIVE
            if index <= 0:
                raise ValueError('Index must be strictly positive')
            # _, mean, dispersion = self.additive_to_reproductive(p, theta, index)
            dispersion = 1. / index
            # Jorg [2], p 131 equation after 4.12
            if p == 1:
                mean = np.exp(theta)
            else:
                mean = ((1 - p) * theta) ** (alpha - 1)

        prefix = ('' if theta == 0 else 'tilted ')
        match p:
            case 0:
                self.name = "normal"
            case 1:
                self.name = "poisson"
            case 2:
                self.name = "gamma"
            case 3:
                if theta == 0:
                    self.name = "levy"
                else:
                    self.name = "inverse gaussian"
            case p if 1 < p < 2:
                self.name = "tweedie"
            case p if p < 0:
                self.name = prefix + "extreme stable"
            case p if p > 2:
                self.name = prefix + "positive extreme stable"
            case p if np.isinf(p):
                self.name = prefix + "extreme cauchy"
            case _:
                raise ValueError(f'Invalid p: {p}')

        # set member variables
        self.p = p
        self.alpha = alpha
        self.mean = mean
        self.dispersion = dispersion
        self.theta = theta
        self.index = index
        # variables prefixed tw_ refer to the created Tweedie distribution
        self.tw_variance = dispersion * mean ** p if self.mode == Mode.REPRODUCTIVE else index * mean ** p
        self.tw_sd = np.sqrt(self.tw_variance)
        self.tw_cv = self.tw_sd / mean
        self._fz = None  # for the frozen distribution object
        self._ft = None

    def dual(self):
        """Create the dual object."""
        match self.mode:
            case Mode.REPRODUCTIVE:
                return Tweedie(self.p, theta=self.theta, index=self.index)
            case Mode.ADDITIVE:
                return Tweedie(self.p, mean=self.mean, dispersion=self.dispersion)

    @staticmethod
    def from_p_mean_cv(p: float, mean: float, cv: float) -> "Tweedie":
        """
        Create a Tweedie object from the parameters p, mean, and the output
        Tweedie cv. The dispersion is computed by equating two expressions
        for the variance:

              (cv * mean) ** 2 = dispersion * mean^p.
          ==>       dispersion = cv **2 * mean ** (2 - p)

        The object is created using p, mean, and dispersion in reproductive
        mode.
        """
        dispersion = cv ** 2 * mean ** (2 - p)
        return Tweedie(p=p, mean=mean, dispersion=dispersion)

    @staticmethod
    def from_po_gamma(frequency: float, *, sev_m: Optional[float] = None, sev_cv: Optional[float] = None,
                      sev_shape: Optional[float] = None, sev_rate: Optional[float] = None,
                      tw_cv: Optional[float] = None) -> "Tweedie":
        """
        Create a Tweedie object for the compound Poisson-gamma case, 1 < p < 2. Inputs

        * frequency: mean number of claims, required
        * One of

            * sev_m: mean severity and sev_cv or
            * sev_shape and sev_rate or
            * sev_m and tw_cv (overall distribution cv)

        Gamma distribution with density f(x) = x^(shape-1) exp(-x rate) / Gamma(shape) for x > 0.

        Higher rate (Poisson rate of emission of particles per unit time) results in a lower mean
        waiting time for n to appear. Gamma(shape, rate) has mean shape / rate and variance
        shape / rate^2. Hence, cv^2 = 1 / shape.

        The object is created using p, mean, and dispersion in reproductive
        mode, same as ``from_p_mean_cv``.
        """
        if sev_m is not None and sev_cv is not None:
            #
            sev_shape = sev_cv ** -2.
            sev_rate = sev_shape / sev_m
        elif sev_shape is not None and sev_rate is not None:
            sev_m = sev_shape / sev_rate
        elif sev_m is not None and tw_cv is not None:
            tw_mean = frequency * sev_m
            tw_var = (tw_mean * tw_cv) ** 2
            # tw_var = freq alpha (alpha + 1) / beta^2 and alpha / beta = sev_m
            # subst for alpha and solve for beta = sev_rate
            sev_rate = tw_mean / (tw_var - frequency * sev_m ** 2)
            sev_shape = sev_m * sev_rate
        else:
            raise ValueError('Must input sev_m and sev_cv, or sev_shape and sev_rate, or sev_m and tw_cv')

        # now figure the Tweedie class reproductive parameters: have all four parameters to
        # play with
        shape = np.array(sev_shape)
        # know that sh=0 have p=2 (gamma), sh=1 have p=1 (Poisson), p >= 0
        # so p = (shape + 2) / (shape + 1)
        p = (shape + 2) / (shape + 1)
        mean = frequency * sev_m
        # variance = mean^p * dispersion = frequency * shape * (shape + 1) / sev_rate^2
        dispersion = frequency * shape * (shape + 1) * sev_rate ** -2 / mean ** p
        return Tweedie(p=p, mean=mean, dispersion=dispersion)

    @staticmethod
    def p_to_alpha(p: float) -> float:
        """Convert p to Jorgensen's alpha, [2] eq 4.9 p 131."""
        # NumPy follows IEEE 754, where division by zero results in np.inf or -np.inf, rather
        # than raising an exception. GPT tells me this is OK practice.
        if np.isinf(p):
            return 1.
        else:
            p = np.array(p)
            return (p - 2) / (p - 1)

    @staticmethod
    def alpha_to_p(alpha: float) -> float:
        """Convert alpha to p, [2] eq 4.10 p 131."""
        alpha = np.array(alpha)
        return (alpha - 2) / (alpha - 1)

    @staticmethod
    def tau(p: float, theta: float) -> float:
        """
        Convert theta (canonical parameter) to mean, using tau = kappa'(theta).

        Evaluate the tau function, [2] below eq 4.12 p 131.
        Recall (1-alpha)(1-p) = -1.
        Assumes dispersion and index = 1, see additive_to_reproductive
        for more general function.
        """
        if 0 < p < 1:
            raise ValueError(f'p cannot be in (0, 1), got {p}')
        if p == 1:
            return jnp.exp(theta)
        elif np.isinf(p):
            # Cauchy, [2] p. 163
            return -jnp.log(-theta) if theta < 0 else 0.
        elif p == 0:
            # normal
            return theta
        elif (p > 2 or p < 0) and theta == 0:
            # valid cases where mean does not exist
            return np.inf
        elif theta == 0:
            raise ValueError(f'Invalid theta=0 for p={p}')
        else:
            # [2] says mu = (theta/(alpha - 1)) ** (alpha - 1)
            # but alpha - 1 = -1 / (p - 1) = 1 / (1 - p)
            # and we have ruled out p = 1
            # a = Tweedie.p_to_alpha(p)
            return ((1 - p) * theta) ** (1 / (1 - p))
                # m2 = (theta / (a - 1)) ** (a - 1)
                # if not np.isinf(p):
                #     if np.isinf(m1) and np.isinf(m2):
                #         pass
                #     else:
                #         assert abs(m1 - m2) < 1e-15, f'Tweedie.tau: {p=}, {theta=}, {m1=}, {m2=}'

    @staticmethod
    def tau_inverse(p: float, mean: float) -> float:
        """
        Convert mean to theta (canonical parameter) using tau_inverse.

        Evaluate the inverse of the tau function, [2] below eq 4.12 p 131.
        Assumes dispersion and index = 1, see reproductive_to_additive
        for more general function.
        In GLMs tau_inverse is called the canonical link function, mapping
        the mean to the canonical parameter.
        """
        if p == 1:
            # poisson, canonical link is log
            return np.log(mean)
        elif np.isinf(p):
            return -np.exp(-mean)
        else:
            # inverse of tau function
            return mean ** (1 - p) / (1 - p)

    @staticmethod
    def additive_to_reproductive(p: float, theta: float, index: float) -> Tuple[float, float, float]:
        """
        Convert additive parameters to reproductive parameters *for the same distribution*.

        Warning: this is not the dual distribution!

        Uses [2] eq 4.8 p. 130.
        """
        # note when p==1 (Poisson), this is still correct
        return (p,
            index * Tweedie.tau(p, theta),
            index ** (1 - p)
            )

    @staticmethod
    def reproductive_to_additive(p: float, mean: float, dispersion: float) -> Tuple[float, float, float]:
        """
        Convert reproductive parameters to additive parameters *for the same distribution*.

        Warning: this is not the dual distribution!

        Uses [2] Exercise 4.5 and equation below 4.8 on p. 130 (that has a typo).
        """
        if p == 1:
            # Poisson
            if dispersion == 1:
                return p, np.log(mean), 1.
            else:
                raise ValueError(f'Poisson additive requires dispersion=1, not {dispersion}')
        else:
            # note slightly evil, dispersion = sigma^2 = 1 / index, so
            # re-enter the square!
            print(f'{p=}, {mean=}, {dispersion=}, {mean * dispersion ** (1 / (p - 1))=}')
            return (p,
                    Tweedie.tau_inverse(p, mean * dispersion ** (1 / (p - 1))),
                    dispersion ** (1 / (1 - p))
                    )

    @property
    def reproductive(self)  -> Tuple[float, float, float]:
        """Return the reproductive parameters: p, mean, dispersion."""
        return self.p, self.mean, self.dispersion

    @property
    def additive(self) -> Tuple[float, float, float]:
        """Return the additive parameters: frequency (Poisson mean), gamma shape and rate."""
        return self.p, self.theta, self.index

    def _po_gamma_conniptions(self):
        """Compute the various Po-gamma compound parameters."""
        # Valid only when 1 < p < 2.
        assert 1 < self.p < 2, "Only valid for 1 < p < 2 distributions."

        # for both reproductive and additive
        shape = (2 - self.p) / (self.p - 1)
        # po-gamma (both modes)
        # mean = freq * shape / rate
        # var = freq * shape * (shape + 1) / rate^2
        match self.mode:
            case Mode.REPRODUCTIVE:
                # mean = self.mean
                # var = self.dispersion * self.mean^p
                rate = (shape + 1) / (self.dispersion * self.mean ** (self.p - 1))
                frequency = self.mean * rate / shape
            case Mode.ADDITIVE:
                # self.mean is computed from p and self.theta
                # mean = self.index * self.mean
                # var = self.index * self.mean^p
                rate = (shape + 1) / (self.index * self.mean ** (self.p - 1))
                frequency = self.index * self.mean  * rate / shape
        # this is just standard gamma distribution conniptions
        sev_m = shape / rate   # noqa
        sev_cv = shape ** -0.5
        # return all options
        return frequency, sev_m, sev_cv, shape, rate  # noqa

    @property
    def po_gamma_m_cv(self) ->  Tuple[float, float, float]:
        """Return Poisson-gamma parameters, specifying gamma using mean and cv."""
        assert 1 < self.p < 2, "Only valid for 1 < p < 2 distributions."
        f, m, cv, sh, rate = self._po_gamma_conniptions()
        return f, m, cv

    @property
    def po_gamma_shape_rate(self) -> Tuple[float, float, float]:
        """Return Poisson-gamma parameters, specifying gamma using shape and rate."""
        assert 1 < self.p < 2, "Only valid for 1 < p < 2 distributions."
        f, m, cv, sh, rate = self._po_gamma_conniptions()
        return f, sh, rate

    @property
    def fz(self):
        """
        Return the frozen distribution object, reproductive or additive depending
        on how the object was initialized.

        Poisson
        --------
        Rather than a scipy.stats object, which requires index=1, use
        an Aggregate object. This allows scaling. Note, for Poisson is
        only reproductive in general. It is

            Y ~ Po(index * exp(theta)) / index

        The Aggregate is also not fussy about pdf vs pmf.

        Gamma
        ------
        Per [2] p. 89 Ga*(theta, lambda) ~ Gamma with shape lambda and
        scale -theta.

        """
        if self._fz is None:
            match self.p:
                case 1:
                    # Poisson or over-dispersed Poisson
                    match self.mode:
                        case Mode.REPRODUCTIVE:
                            # ODP, needs to be an aggregate object, Y = Z / index
                            mean = self.index * self.mean
                            temp = ss.poisson(mean)
                            mx = temp.isf(1e-12)
                            log2 = int(np.ceil(np.log2(mx))) + 1
                            assert log2 <= 20, f'log2 too large: {log2}'
                            # build agg object to include scaling
                            self._fz = build(f'agg Po{id(self)} {mean} claims dsev [{1 / self.index}] poisson', update=False)
                            # update with appropriate bs and log2
                            self._fz.update(bs=1 / self.index, log2=log2)
                        case Mode.ADDITIVE:
                            self._fz = ss.poisson(self.index * self.mean)
                            # top unnecessary complaining
                            self._fz.pdf = self._fz.pmf
                case 2:
                    # gamma
                    match self.mode:
                        case Mode.REPRODUCTIVE:
                            self._fz = ss.gamma(self.index, scale=-1 / (self.theta * self.index))
                        case Mode.ADDITIVE:
                            self._fz = ss.gamma(self.index, scale=-1 / self.theta)
                case 3:
                    # inverse Gaussian or Levy Stable 1/2
                    match self.mode:
                        case Mode.REPRODUCTIVE:
                            if self.theta < 0:
                                self._fz = ss.invgauss(self.mean / self.index, scale=self.index)
                            else:
                                # has no shape parameters
                                self._fz = ss.levy(scale=self.index ** (2. - 1.))
                        case Mode.ADDITIVE:
                            if self.theta < 0:
                                self._fz = ss.invgauss(self.mean / self.index, scale=self.index ** 2)
                            else:
                                # has no shape parameters
                                self._fz = ss.levy(scale=self.index ** 2.)
                case 0:
                    # normal
                    match self.mode:
                        case Mode.REPRODUCTIVE:
                            self._fz = ss.norm(loc=self.mean, scale=1.)
                        case Mode.ADDITIVE:
                            self._fz = ss.norm(loc=self.mean, scale=self.dispersion ** 0.5)
                case self.p if 1 < self.p < 2:
                    # Tweedie distribution
                    # get frequency and gamma parameters
                    # po_gamma_shape_rate accounts for mode of self.
                    f, s, r = self.po_gamma_shape_rate
                    self._fz = build(f'agg TW{id(self)} {f} claims sev {1/r} * gamma {s} poisson')
                case self.p if 2 < self.p < np.inf:
                    # positive extreme stable
                    assert 0 < self.alpha < 1
                    if self.theta < 0:
                        NotImplementedError('Negative theta for positive extreme stable')
                    else:
                        match self.mode:
                            case Mode.REPRODUCTIVE:
                                self._fz = ss.levy_stable(self.alpha, 1.0, loc=0, scale=self.index ** (1 / self.alpha - 1))
                            case Mode.ADDITIVE:
                                self._fz = ss.levy_stable(self.alpha, 1.0, loc=0, scale=self.index ** (1 / self.alpha))
                case self.p if np.isinf(self.p):
                    # Cauchy
                    if self.theta != 0:
                        NotImplementedError('Negative theta for Cauchy; no closed form expression, create via FourierTools')
                    else:
                        print('set up Cauchy')
                        self._fz = ss.levy_stable(self.alpha, 1.0, loc=0, scale=self.index)
                    assert self.alpha == 1
                    match self.mode:
                        case Mode.REPRODUCTIVE:
                            pass
                        case Mode.ADDITIVE:
                            pass
                case self.p if self.p < 0:
                    # extreme stable, supported on whole real line
                    assert 1 < self.alpha < 2
                    # assert self.theta <= 0, f'Invalid theta for spectrally negative extreme stable'
                    # if self.theta < 0:
                    #     NotImplementedError('Negative theta for spectrally negative extreme; no closed form expression, create via FourierTools')
                    # return this regardless of theta (and mode)!
                    self._fz = ss.levy_stable(self.alpha, -1.0, scale=self.index ** (1 / self.alpha))
                    # match self.mode:
                    #     case Mode.REPRODUCTIVE:
                    #         # raise NotImplementedError('Reproductive spectrally positive extreme stable')
                    #     case Mode.ADDITIVE:
                    #         self._fz = ss.levy_stable(self.alpha, -1.0, scale=self.index ** (1 / self.alpha))
                case _:
                    # this should really be caught elsewhere an never occur!
                    raise ValueError(f'Invalid p: {self.p}')
        return self._fz

    @property
    def ft(self):
        """Create and return FourierTools object."""
        if self._ft is None:
            # scale_mode false because chf already incorporates scale
            self._ft = FourierTools(self.chf, self.fz, scale_mode=False)
        return self._ft

    def stats(self):
        """Empirical moments of the distribution."""
        if isinstance(self.fz, Aggregate):
            return self.fz.agg_m, self.fz.agg_var, self.fz.agg_skew
        elif self.fz:
            return self.fz.stats('mvs')
        else:
            return np.array([np.nan]*3)

    def audit(self):
        """
        Audit dataframe comparing definition, Fourier, and frozen stats.

        Moment code from aggregate xsden_to_meancvskew.
        """
        if self.ft._df is None:  # noqa
            raise ValueError('Must compute before estimating stats!')
        xs = np.array(self.ft.df.index)
        den = self.ft.df.p.values
        pg = 1 - den.sum()
        xd = xs * den
        if isinstance(xs, np.ndarray):
            xsm = xs[-1]
            bs = xs[1] - xs[0]
        elif isinstance(xs, pd.Series):
            xsm = xs.iloc[-1]
            bs = xs.iloc[1] - xs.iloc[0]
        else:
            _ = np.array(xs)
            xsm = _[-1]
            bs = _[1] - _[0]
        xsm = xsm + bs
        ex1 = np.sum(xd) + pg * xsm
        xd *= xs
        ex2 = np.sum(xd) + pg * xsm ** 2
        ex3 = np.sum(xd * xs) + pg * xsm ** 3
        vf = ex2 - ex1 ** 2
        # convert to cv and skew
        mw = MomentWrangler()
        mw.noncentral = ex1, ex2, ex3
        me, cve, ske = mw.mcvsk
        m, v, sk = self.stats()
        cv = v ** .5 / m
        # moments from the definition
        if self.mode == Mode.REPRODUCTIVE:
            am = self.mean
            va = self.dispersion * self.mean ** self.p
        else:
            am = self.mean * self.index
            va = self.index * self.mean ** self.p
        cva = va ** .5 / am
        # skewness IDLE grad
        f = lambda x: self.tau(self.p, x)
        if self.mode == Mode.REPRODUCTIVE:
            ska = grad(grad(f))(self.theta) * self.index ** -2 / self.tw_variance ** 1.5
        elif self.mode == Mode.ADDITIVE:
            ska = grad(grad(f))(self.theta) * self.index / self.tw_variance ** 1.5
        # assemble answer
        actual = 'reproductive' if self.mode == Mode.REPRODUCTIVE else 'additive'
        ans = pd.DataFrame({actual: [am, va, cva, ska], 'fourier': [me, vf, cve, ske], 'frozen': [m, v, cv, sk]},
                          index=['EX', 'var', 'CV', 'Skew'])
        ans['abs err'] = (ans.fourier - ans.frozen).abs()
        ans['rel err'] = ans['abs err'] / ans['frozen']
        return ans

    def kappa(self, theta: float):
        """
        Cumulant function of the carrier distribution.

        Jorgensen [2] equation 4.13, p. 131.
        """
        if self.p == 1:
            return np.exp(theta)
        elif self.p == 2:
            return -np.log(-theta)
        elif np.isinf(self.p):
            # [2] p. 163
            # print(theta - theta * np.log(-theta))
            return np.where(theta==0, 0,
                            theta - theta * np.log(-theta))
        else:
            # per above case are guaranteed alpha != 1
            assert self.alpha != 1, "WTF??"
            return ((self.alpha - 1) / self.alpha *
                    (theta / (self.alpha - 1)) ** self.alpha)

    def K_additive(self, s):
        """
        Cumulant function, reflecting theta and in additive form.

         See [2] eq. 3.4 and eq 4.15.
         """
        return self.index * (self.kappa(self.theta + s) - self.kappa(self.theta))

    def K_reproductive(self, s):
        """
        Cumulant function, reflecting theta and in additive form.

        See, [2] eq. 3.5 and eq 4.15.
        """
        return self.index * (self.kappa(self.theta + s / self.index) - self.kappa(self.theta))

    def K(self, s):
        """Cumulant function, reflecting how the class was initialized."""
        match self.mode:
            case Mode.REPRODUCTIVE:
                return self.K_reproductive(s)
            case Mode.ADDITIVE:
                return self.K_additive(s)

    # def K_manual(self, s):
    #     """
    #     Cumulant function, reflecting theta and in the additive form.
    #
    #     K(s) = lambda * (kappa(theta + s) - kappa(theta))
    #
    #     Per Jorg [2] p. 132 Eq 4.15. Note special treatment
    #     of theta = 0, since 4.15 divides by theta.
    #     Belt 'n braces tester: the same as K!
    #     """
    #     if self.theta == 0:
    #         assert self.p != 2, 'p=2 does not allow theta = 0'
    #         return self.index * (self.kappa(s) - self.kappa(0))
    #     assert self.index != 0, 'unexpected, index cannot be 0'
    #     # functions are now scaled by index
    #     if self.p == 1:
    #         return self.index * np.exp(self.theta) * (np.exp(s) - 1)
    #     elif self.p == 2:
    #         return -self.index * np.log(1 + s / self.theta)
    #     else:
    #         # p != 1, 2
    #         return self.index * self.kappa(self.theta) * ((1 + s / self.theta) ** self.alpha - 1)

    def chf(self, t):
        """Characteristic function."""
        return np.exp(self.K(1j * t))

    def levy_density(self, x):
        """Levy density."""
        raise NotImplementedError('Levy density not implemented')

    @staticmethod
    def reproductive_to_tweedie_compound(p, mean, dispersion):
        """
        Convert Tweedie reproductive parameters to additive parameters for the compound
        Poisson-gamma distribution.

        This is the inverse of the Poisson-gamma conversion.
        """
        if not (1 < p < 2):
            raise ValueError(f'Only valid for 1 < p < 2 distribution, not {p=}')
        var = dispersion * mean ** p
        shape = (2 - p) / (p - 1)
        beta = var / (shape + 1)
        frequency = mean / (shape * beta)
        return frequency, shape, beta

    def to_series(self):
        """Convert to Series."""
        mode = 'reproductive' if self.mode == Mode.REPRODUCTIVE else 'additive'
        ans = pd.Series([mode, repr(self), self.p, self.mean, self.dispersion, self.theta, self.index, self.tw_cv],
                        index=pd.Index(['mode', 'key', 'p', 'mean', 'dispersion', 'theta', 'index',
                                        'tw_cv'],
                                       name='quantity'),
                        name='value')
        return ans

    def to_series_ex(self):
        """As to_series but includes Tweedie compound parameters."""
        if not (1 < self.p < 2):
            raise ValueError(f'Only valid for 1 < p < 2 distribution, not {self.p=}')
        frequency, sev_m, sev_cv, shape, rate  = self._po_gamma_conniptions()
        ans2 = pd.Series([frequency, sev_m, sev_cv, shape, rate],
                         index=['frequency', 'sev_m', 'sev_cv', 'shape', 'rate'],
                         name='quantity')
        return pd.concat([self.to_series(), ans2])

    def to_tex(self):
        df = self.to_frame().rename(index=self._renamer)
        df['parameterization'] = ['Reproductive'] * 3 + ['Additive'] * 2  + ['Tw stats']
        df = df.reset_index(drop=False).set_index(['parameterization', 'quantity'])
        return df

    def to_frame(self):
        """Convert to DataFrame."""
        return self.to_series().to_frame()

    def to_json(self):
        return self.to_series().to_dict()

    def to_decl(self, name='TW'):
        """Parameters needed to feed to Aggregate using tweedie keyword."""
        if self.mode == Mode.REPRODUCTIVE:
            return f'agg {name} tweedie {self.mean} {self.p} {self.dispersion}'
        else:
            raise NotImplementedError('Additive decl not yet implemented')

    def __repr_html__(self):
        return self.to_frame().to_html()

    def __repr__(self):
        if self.mode == Mode.ADDITIVE:
            return f'Tweedie*(p={self.p:.3g}, θ={self.theta:.3g}, λ={self.index:.3g})'
        elif self.mode == Mode.REPRODUCTIVE:
            return f'Tweedie(p={self.p:.3g}, μ={self.mean:.3g}, σ^2={self.dispersion:.3g}; θ={self.theta:.3g})'
        else:
            return f'WTF is that...mode={self.mode}'

    # def __str__(self):
    #     star = '*' if self.mode == Mode.ADDITIVE else ''
    #     return f'Tweedie{star}(p={self.p:.3g}, μ={self.mean:.3g}, sigma2={self.dispersion:.3g}, θ={self.theta:.3g})'

    def _repr_mimebundle_(self, include=None, exclude=None):  # noqa
        return {
            # "application/json": {"key": "value"},
            "text/html": self.to_frame().to_html(),
            "text/plain": repr(self)
        }

    def test(self, bs=None, x_min=0, log2=16, s=1e-8, plot=True, simpson=False):
            """Test the Tweedie class object."""
            n = 1 << log2
            if bs is None:
                bs = (self.ft.fz.ppf(1-s) - x_min) / n
                bs = 2 ** np.round(np.log2(bs), 0)
                print(f'{bs=}, {1/bs=}')
            self.ft.invert(log2, x_min=x_min, bs=bs)
            if self.ft.fz is not None:
                self.ft.compute_exact()
            if plot:
                self.ft.plot(suptitle=self.name, verbose=False)
                # for ax in self.ft.last_fig.axes[:2]:
                #     l, u = ax.get_xlim()
                #     ax.set(xlim=[-u/20, u])
            else:
                if plot:
                    self.ft.df.plot(figsize=(3, 2.25))
            if simpson:
                self.ft.invert_simpson(log2=log2, x_min=x_min, bs=bs)
                ax = self.ft.df.plot(figsize=(3,2.25), logy=True, title='2 methods')
                if self.ft.fz is not None:
                    self.ft.df_exact.plot(ax=ax)
            return self.audit()


def make_test_suite(mode=Mode.REPRODUCTIVE):
    """Create a test suite for the Tweedie class."""
    p = Path(r'C:\Users\steve\S\TELOS\Blog\quarto\ConvexConsiderations\posts\notes\2025-02-06-Tweedie-distributions\test-cases.csv')
    assert p.exists()
    tests = pd.read_csv(p)
    # convert str to list:
    tests.theta = tests.theta.map(eval)
    tests['index'] = tests['index'].map(eval)
    # explode lists
    tests_ex = tests.explode('theta').explode('index').reset_index(drop=True)
    tests_ex.index.name = 'n'
    # convert to floats
    for c in ['p', 'alpha', 'theta', 'index']:
        tests_ex[c] = [eval(i) if type(i)==str else float(i)  for i in tests_ex[c]]

    # figure reproductive parameters: tau is not vectorized
    tests_ex['mu'] = [Tweedie.tau(p, t) for p, t in tests_ex[['p', 'theta']].values]
    tests_ex['dispersion'] = 1 / tests_ex['index']

    # figure expected repro moments
    tests_ex['mean_reproductive'] = tests_ex['mu']
    tests_ex['variance_reproductive'] = tests_ex['mu'] ** tests_ex['p'] * tests_ex['dispersion']
    tests_ex['sd'] = tests_ex.variance_reproductive ** 0.5
    tests_ex['cv'] = tests_ex.sd / tests_ex['mean_reproductive']
    tests_ex = tests_ex.drop(columns=['sd', 'notes'])

    # expected additive moments
    tests_ex['mean_additive'] = tests_ex['mu'] * tests_ex['index']
    tests_ex['var_additive'] = tests_ex['index'] * tests_ex['mu'] ** tests_ex['p']
    tests_ex['cv_additive'] = tests_ex.var_additive ** 0.5 / tests_ex.mean_additive
    return tests_ex


def run_test(p, theta, index, log2=16, bs=None, s=1e-12, **kwargs):
    """Run a test of the Tweedie class."""
    # additive
    ta = Tweedie(p, theta=theta, index=index)
    print(ta)
    ans = ta.test(bs=bs, log2=log2, s=s, **kwargs)
    if 1 < p < 2:
        f = ta.po_gamma_shape_rate
        print(f, ta.po_gamma_m_cv, sep='\n')
        print(np.exp(-f[0]))
    display(ans)
    print('='*80)
    # gamma reproductive ====================================
    tr = ta.dual()
    print(tr)
    ans = tr.test(bs=bs, log2=log2, s=s, **kwargs)
    if 1 < p < 2:
        f = tr.po_gamma_shape_rate
        print(f, tr.po_gamma_m_cv, sep='\n')
        print(np.exp(-f[0]))
    display(ans)
    return ta, tr


def tweedie_illustration():
    """Make the usual graph of Tweedie distributions."""
    # redo with Jorg sign for alpha
    alpha = np.linspace(-4, 2, 101)
    p = (-alpha+2) / (-alpha+1)

    f, ax = plt.subplots(figsize=(5,5))
    # want to cut out the vertical line and color code by type of distribution
    idx = (1 <= p) * (p < 2)
    ax.plot(alpha[idx], p[idx], lw=3, c='C0')
    idx = 2 <= p
    ax.plot(alpha[idx], p[idx], lw=3, c='C1')
    idx = p <= 0
    ax.plot(alpha[idx], p[idx], lw=3, c='C2')

    # asymptotes
    ax.axhline(1, c='g', lw=.5, ls='--')
    ax.axhline(2, c='g', lw=.5, ls='--')
    ax.axvline(1, c='g', lw=.5, ls='--')

    # axes
    ax.axvline(0, c='k', lw=.25)
    ax.axhline(0, c='k', lw=.25)

    # ticks and axis labels
    ax.set(xlabel='$\\alpha=(p-2)/(p-1)$,  jump tail density $\\propto e^{\\theta x}/x^{\\alpha + 1}$', #  base jump density is $x^{\\bar\\alpha}$',
           ylabel='Variance power function $p$, $V(\\mu)=\\mu^p$, $p=(\\alpha-2)/(\\alpha-1)$',
           ylim=[-5,10.3], xlim=[-5., 2.5])
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(-4,11,1)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}')) #  if int(x)%2==0 or x==3 else '' ))
    # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(-4,10,0.5)))
    # ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(-2,3,0.5)))

    # annotations
    def ql(x, y, t, dot=True, ha='right', va='bottom'):
        """Handy annotator function."""
        voff = {'bottom': 0.2, 'top': -.2, 'center': 0.}
        hoff = {'left': 0.2, 'right': -0.2, 'center': 0.}
        ax.text(x + hoff[ha], y + voff[va], t, ha=ha, va=va)
        if dot:
            ax.plot(x,y, 'ko', ms=5)

    ql(2,    0, 'Normal', ha='center')
    ql(1,   10, 'Cauchy\nas $p\\to\\infty$', ha='left', va='top')
    ql(.5,   3, 'Levy stable 3/2\ninv Gaussian', ha='right')
    ql(0,    2, 'Gamma')
    ql(-4.5, 1, 'Poisson\n$\\alpha\\to-\\infty$', va='top', ha='center')

    ql(1.25, -3, 'Extreme\nstable,\n$\\theta ≥ 0$', False, ha='left', va='center')
    ql(0.5, -3, 'Positive\nextreme\nstable,\n$\\theta ≤ 0$', False, ha='center', va='center')
    ql(-2,  -3, 'Tweedie,\n$\\theta < 0$', False, ha='center', va='center')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)