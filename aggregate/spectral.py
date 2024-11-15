from collections.abc import Iterable
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from io import StringIO
import pandas as pd
# from textwrap import fill
import logging
import numba

from .constants import *
from .random_agg import RANDOM
from .utilities import short_hash

logger = logging.getLogger(__name__)


# number super-fast functions for TVaR and BiTVaR computations
@numba.njit(parallel=False)
def tvar_gS(probs, p):
    """
    Compute gS given individual probs. Equivlaent to computing
    ``tvar.g(1 - probs.cumsum())`` for a tvar distortion.

    Runs about 4x speed of g.g(1-probs.cumsum()) for TVaR. Cannot
    parallelize because of shared S variable.

    :param probs: numpy array of probabilities.
    :param p: float, distortion parameter.
    """
    # shared S variable mean you cannot parallelize
    S = 1.
    ans = np.zeros_like(probs)

    if p == 0:
        # mean
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S >= 1:
                ans[i] = 1
            else:
                ans[i] = S

    elif p == 1:
        for i in range(len(probs)):
            S -= probs[i]
            if S > 0:
                ans[i] = 1
            else:
                ans[i] = 0

    else:
        # proper tvar
        s = 1 - p
        m = 1 / s
        for i in range(len(probs)):
            S -= probs[i]
            if S < s:
                ans[i] = m * S
            else:
                ans[i] = 1
    return ans


@numba.njit(parallel=False)
def bitvar_gS(probs, p0, p1, w):
    """
    Compute gS given individual probs. Equivlaent to computing
    ``bitvar.g(1 - probs.cumsum())`` for a bitvar distortion.

    As usual, p0 < p1, w in (0,1) input parameters.
    See OneNote 2024/November Numba TVaR and BiTVaR.

    :param probs: numpy array of probabilities.
    :param p0: float, distortion parameter.
    :param p1: float, distortion parameter.
    :param w: float, weighting to p1.
    """

    if p0 == p1:
        # actually a tvar
        return tvar_gS(probs, p0)

    if w == 0:
        return tvar_gS(probs, p0)

    if w == 1:
        return tvar_gS(probs, p1)

    # now weight is 0 < w < 1
    # now there are four cases (0,1), (>0, 1), (0, <1), (>0, <1)
    s0 = 1 - p0
    s1 = 1 - p1
    S = 1.
    ans = np.zeros_like(probs)
    # height at kink
    pt = w + (1 - w) * s1 / s0
    if p0 == 0 and p1 == 1:
        slope = (1 - w)
        for i in range(len(probs)):
            S -= probs[i]
            # here and below say <= 0 to allow for
            # numerical cumulation error which can result
            # in S very small < 0
            if S <= 0:
                ans[i] = 0
            else:
                ans[i] = w + slope * S

    elif p0 > 0 and p1 == 1:
        slope = (1 - w) / s0
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S < s0:
                ans[i] = w + slope * S
            else:
                ans[i] = 1.

    elif p0 == 0 and p1 < 1:
        # two kinks
        slope0 = pt / s1
        slope1 = (1 - pt) / p1
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S < s1:
                ans[i] = slope0 * S
            else:
                ans[i] = pt + slope1 * (S - s1)

    else:
        # p0 > 0 and p1 < 1
        # three kinks
        slope0 = pt / s1
        slope1 = (1 - pt) / (s0 - s1)
        for i in range(len(probs)):
            S -= probs[i]
            if S <= 0:
                ans[i] = 0
            elif S < s1:
                ans[i] = slope0 * S
            elif S < s0:
                ans[i] = pt + slope1 * (S - s1)
            else:
                ans[i] = 1.

    return ans


@numba.njit(parallel=False)
def tvar_ra(probs, x, p):
    """
    Computes risk adjusted value in one step with no intermediate
    arrays or copies.

    Equivalent to::

        dx = np.diff(x)
        (g.g(1 - probs.cumsum())[:-1] * dx).sum() + x[0]

    or to::

        den = pd.Series(probs, index=x)
        tvar.price_ex(den, kind='both', calculation='dx', as_frame=True)


    Detects if it is computing the  mean (p==0) or the maximum (p==1)
    and shortcuts accordingly.

    :param probs: numpy array of probabilities.
    :param x: numpy array loss outcomes, must be in ascending order.
    :param p: float, distortion parameter.
    """
    # shared S variable mean you cannot parallelize
    S = 1.
    lastx = x[0]
    ans = lastx    # !! remember the shift!

    if p == 0:
        # mean
        ans = np.sum(probs * x)
        # for i in range(len(probs) - 1):
        #     S -= probs[i]
        #     dx = x[i + 1] - lastx
        #     lastx = x[i + 1]
        #     if S <= 0:
        #         pass
        #     elif S >= 1:
        #         ans += dx
        #     else:
        #         ans += S * dx

    elif p == 1:
        # max
        ans = x[-1]
        # for i in range(len(probs) - 1):
        #     S -= probs[i]
        #     dx = x[i + 1] - lastx
        #     lastx = x[i + 1]
        #     if S > 0:
        #         ans += dx

    else:
        # proper tvar
        s = 1 - p
        m = 1 / s
        for i in range(len(probs) - 1):
            S -= probs[i]
            dx = x[i + 1] - lastx
            lastx = x[i + 1]
            if S < s:
                ans += m * S * dx
            else:
                ans += dx
    return ans


@numba.njit(parallel=False)
def bitvar_ra(probs, x, p0, p1, w):
    """
    Computes risk adjusted value in one step with no intermediate
    arrays or copies.

    Equivalent to::

        dx = np.diff(x)
        (g.g(1 - probs.cumsum())[:-1] * dx).sum() + x[0]

    or to::

        den = pd.Series(probs, index=x)
        tvar.price_ex(den, kind='both', calculation='dx', as_frame=True)

    :param probs: numpy array of probabilities.
    :param x: numpy array of loss outcomes, must be in ascending order.
    :param p0: float, distortion parameter.
    :param p1: float, distortion parameter.
    :param w: float, weighting to p1.
    """

    if p0 == p1:
        # actually a tvar
        return tvar_ra(probs, x, p0)

    if w == 0:
        return tvar_ra(probs, x, p0)

    if w == 1:
        return tvar_ra(probs, x, p1)

    # now weight is 0 < w < 1
    # now there are four cases (0,1), (>0, 1), (0, <1), (>0, <1)
    s0 = 1 - p0
    s1 = 1 - p1
    S = 1.
    lastx = x[0]
    ans = lastx
    # height at kink
    pt = w + (1 - w) * s1 / s0

    if p0 == 0 and p1 == 1:
        # weights mean and max
        m = np.sum(probs * x)
        x = x[-1]
        ans = w * x + (1 - w) * m
        # slope = (1 - w)
        # for i in range(len(probs) - 1):
        #     S -= probs[i]
        #     dx = x[i + 1] - lastx
        #     lastx = x[i + 1]
        #     # here and below say <= 0 to allow for
        #     # numerical cumulation error which can result
        #     # in S very small < 0
        #     if S <= 0:
        #         pass
        #     else:
        #         ans += (w + slope * S) * dx

    elif p0 > 0 and p1 == 1:
        m0 = tvar_ra(probs, x, p0)
        ans = w * x[-1] + (1 - w) * m0
        # slope = (1 - w) / s0
        # for i in range(len(probs) - 1):
        #     S -= probs[i]
        #     dx = x[i + 1] - lastx
        #     lastx = x[i + 1]
        #     if S <= 0:
        #         pass
        #     elif S < s0:
        #         ans += (w + slope * S) * dx
        #     else:
        #         ans += dx

    elif p0 == 0 and p1 < 1:
        m1 = tvar_ra(probs, x, p1)
        ans = w * m1 + (1 - w) * np.sum(probs * x)
        # two kinks
        # slope0 = pt / s1
        # slope1 = (1 - pt) / p1
        # for i in range(len(probs) - 1):
        #     S -= probs[i]
        #     dx = x[i + 1] - lastx
        #     lastx = x[i + 1]
        #     if S <= 0:
        #         pass
        #     elif S < s1:
        #         ans += (slope0 * S) * dx
        #     else:
        #         ans += (pt + slope1 * (S - s1)) * dx

    else:
        # p0 > 0 and p1 < 1
        # three kinks
        slope0 = pt / s1
        slope1 = (1 - pt) / (s0 - s1)
        for i in range(len(probs) - 1):
            S -= probs[i]
            dx = x[i + 1] - lastx
            lastx = x[i + 1]
            if S <= 0:
                pass
            elif S < s1:
                ans += (slope0 * S) * dx
            elif S < s0:
                ans += (pt + slope1 * (S - s1)) * dx
            else:
                ans += dx

    return ans


class Distortion(object):
    """
    Creation and management of distortion functions.

    0.9.4: renamed roe to ccoc, but kept creator with roe for backwards compatibility.
    Oct 2022: renamed wtdtvar to bitvar, but kept ...
    Sep 2024: added a proper implementation of wtdtvar and a random_distortion using it

    Note, to create a fake Distortion use a type:

        g = type('Distortion', (),
             {'g': your function,
             'g_inv': your function,
             'g_dual': lambda x: 1 - g.g(1-x)}
             )

    """

    # make these (mostly) immutable...avoid changing by mistake
    _available_distortions_ = ('ph', 'wang', 'cll', 'lep', 'ly', 'clin', 'dual', 'ccoc', 'tvar',
                               'bitvar', 'wtdtvar', 'convex', 'tt', 'beta')
    _has_mass_ = ('ly', 'clin', 'lep', 'roe')
    _med_names_ = ("Prop Hzrd", "Wang", 'Capd Loglin', "Lev Equiv", "Lin Yield", "Capped Linear", "Dual Mom",
                   'Const CoC', "Tail VaR", 'BiTVaR', 'WtdTVaR', "Convex Env", 'Wang-tt', 'Beta')
    _long_names_ = ("Proportional Hazard", "Wang-normal", 'Capped Loglinear', "Leverage Equivalent Pricing",
                    "Linear Yield", "Capped Linear", "Dual Moment", "Constant CoC", "Tail VaR",
                    'BiTVaR', 'Weighted TVaR', "Convex Envelope", 'Wang-tt', 'Beta')
    # TODO fix examples!
    # _available_distortions_ = ('ph', 'wang', 'cll', 'lep',  'ly', 'clin', 'dual', 'ccoc', 'tvar', 'wtdtvar,  'convex')
    _eg_param_1_ =              (.9,     .1,      .9,    0.25,  0.8,   1.1,   1.5,   .1,     0.15,     .15)  # noqa
    _eg_param_2_ =              (.5,     .75,     .5,    0.5,   1.5,   1.8,     3,   .25,    0.50,      .5)  # noqa
    # _distortion_names_ = dict(zip(_available_distortions_, _med_names_))
    _distortion_names_ = dict(zip(_available_distortions_, _long_names_))
    renamer = _distortion_names_

    @staticmethod
    def tvar_terms(p_in):
        """
        Evaluate tvar function min(s / (1-p), 1),
        allowing for possibility p=1 and using vector input.
        s = 1 - p in reverse order. This evaluates
        the "knot" points needed to create a general weighted
        TVaR.
        """
        n = len(p_in)
        p = p_in.reshape((n, 1))
        s = (1 - p_in[::-1]).reshape((1, n))
        return np.where(s==0,
                        np.zeros_like(p),
                        np.where(p==1,
                                 np.ones_like(p),
                                 np.minimum(s / (1 - p), 1)))

    @classmethod
    def available_distortions(cls, pricing=True, strict=True):
        """
        List of the available distortions.

        :param pricing: only return list suitable for pricing, excludes tvar and convex
        :param strict: only include those without mass at zero  (pricing only)
        :return:
        """

        if pricing and strict:
            return tuple((i for i in cls._available_distortions_[:-5] if i not in cls._has_mass_))
        elif pricing:
            return cls._available_distortions_[:-3]
        else:
            return cls._available_distortions_

    # Custom __getstate__ to save named constructor arguments
    def __getstate__(self):
        # Save the constructor arguments in a state dictionary with names
        # run through shape
        if isinstance(self.shape, Iterable):
            loc_shape = []
            for x in self.shape:
                if isinstance(x, Distortion):
                    loc_shape.append(x.__getstate__())
                else:
                    loc_shape.append(x)
        else:
            loc_shape = self.shape

        state = {
            'name': self._name,
            'shape': loc_shape,
            'r0': self.r0,
            'df': self.df,
            'col_x': self.col_x,
            'col_y': self.col_y,
            'display_name': self.display_name
        }
        return state

    def __setstate__(self, state):
        # Unpack the state dictionary and pass it to the constructor
        # Use __dict__.update() to restore the state...usual but doesn't work
        # because of possible iteration in shape
        # self.__dict__.update(state)
        self._name = state['name']
        self.r0 = state['r0']
        self.df = state['df']
        self.col_x = state['col_x']
        self.col_y = state['col_y']
        self.display_name = state['display_name']
        # deal with complex shape
        if isinstance(state['shape'], Iterable):
            loc_shape = []
            for x in state['shape']:
                if isinstance(x, dict):
                    loc_shape.append(Distortion(**x))
                else:
                    loc_shape.append(x)
            self.shape = np.array(loc_shape)
        else:
            self.shape = state['shape']
        # now complete the init
        self._complete_init()

    def __init__(self, name, shape, r0=0.0, df=None, col_x='', col_y='', display_name=''):
        """
        Create a new distortion.

        For the beta distribution:

        * A synthesis of risk measures for capital adequacy
        * Wirch and Hardy, IME 1999
        * 0<a<=1, b>= 1
        * b=1 is a PH with rho = 1/a
        * a=1 is a dual with rho = b


        Tester: ::

            ps = np.linspace(0, 1, 201)
            for dn in agg.Distortion.available_distortions(True):
                if dn=='clin':
                    # shape param must be > 1
                    g_dist = agg.Distortion(**{'name': dn, 'shape': 1.25, 'r0': 0.02, 'df': 5.5})
                else:
                    g_dist = agg.Distortion(**{'name': dn, 'shape': 0.5, 'r0': 0.02, 'df': 5.5})
                g_dist.plot()
                g = g_dist.g
                g_inv = g_dist.g_inv

                df = pd.DataFrame({'p': ps, 'gg_inv': g(g_inv(ps)), 'g_invg': g_inv(g(ps)),
                'g': g(ps), 'g_inv': g_inv(ps)})
                print(dn)
                print("errors")
                display(df.query(' abs(gg_inv - g_invg) > 1e-5'))

        :param name: name of an available distortion, call ``Distortion.available_distortions()`` for a list
        :param shape: float or [float, float] for beta
        :param shape: shape parameter
        :param r0: risk free or rental rate of interest
        :param df:  for convex envelope, dataframe with col_x and col_y used to parameterize or df for t
        :param col_x:
        :param col_y:
        :param display_name: over-ride name, useful for parameterized convex fix distributions
        """
        # constructor arguments (e.g., needed for pickling)
        self._name = name
        self.r0 = r0
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
        self.display_name = display_name
        self.shape = shape
        self.min_index = None  # used for minimum distortion, returns "active" distortion index
        self._complete_init()

    def _complete_init(self):
        """

        """
        # when created by calibrate distortions extra info put here
        self.error = 0.0
        self.premium_target = 0.0
        self.assets = 0.0
        self.mass = 0.0
        g_prime = None

        # now make g and g_inv
        if self._name == 'ph':
            rho = self.shape
            rhoinv = 1.0 / rho   # noqa
            self.has_mass = False

            # @numba.vectorize(["float64(float64)"], nopython=True, target='parallel')
            def g(x):
                return x ** rho

            def g_inv(x):
                return x ** rhoinv

            def g_prime(x):
                return np.where(x > 0, rho * x ** (rho - 1.0), np.inf)

        elif self._name == 'beta':
            a, b = self.shape
            assert 0 < a <= 1, f'a parameter must be in (0,1], not {a}'
            assert b >= 1, f'b parameter must be >= 1, not {b}'
            fz = ss.beta(a, b)
            self.has_mass = False

            def g(x):
                return fz.cdf(x)

            def g_inv(x):
                return fz.ppf(x)

            def g_prime(x):
                return fz.pdf(x)

        elif self._name == 'wang':
            lam = self.shape
            n = ss.norm()
            self.has_mass = False

            def g(x):
                return n.cdf(n.ppf(x) + lam)

            def g_inv(x):
                return n.cdf(n.ppf(x) - lam)

            def g_prime(x):
                return n.pdf(n.ppf(x) + lam) / n.pdf(n.ppf(x))

        elif self._name == 'tt':
            lam = self.shape
            t = ss.t(self.df)
            self.has_mass = False

            def g(x):
                return t.cdf(t.ppf(x) + lam)

            def g_inv(x):
                return t.cdf(t.ppf(x) - lam)

        elif self._name == 'cll':
            # capped log linear
            b = self.shape
            binv = 1 / b
            ea = np.exp(self.r0)
            a = self.r0
            self.has_mass = False

            def g(x):
                return np.where(x==0, 0, np.minimum(1, ea * x ** b))

            def g_inv(x):
                return np.where(x < 1, np.minimum(1, (x / ea) ** binv), 1)

        elif self._name == 'tvar':
            p = self.shape
            if p == 1:
                # aka max
                alpha = np.nan
                self.has_mass = True
                self.mass = 1

                def g(x):
                    # <= 0 handles rounding issues gracefully
                    return np.where(x<=0, 0, 1)

                def g_inv(x):
                    return np.where(x < 1, x * (1 - p), 1)

                def g_prime(x):
                    return np.where(x < 1 - p, alpha, 0)

            else:
                # proper tvar or mean
                alpha = 1 / (1 - p)
                self.has_mass = False

                def g(x):
                    return np.minimum(alpha * x, 1)

                def g_inv(x):
                    return np.where(x < 1, x * (1 - p), 1)

                def g_prime(x):
                    return np.where(x <= 1 - p, alpha, 0)


        elif self._name == 'ly':
            # linear yield
            # r0 = occupancy; rk = consumption specified in list shape parameter
            rk = self.shape
            self.has_mass = (self.r0 > 0)
            self.mass = self.r0 / (1 + self.r0)

            def g(x):
                return np.where(x == 0, 0, (self.r0 + x * (1 + rk)) / (1 + self.r0 + rk * x))

            def g_inv(x):
                return np.maximum(0, (x * (1 + self.r0) - self.r0) / (1 + rk * (1 - x)))

        elif self._name == 'clin':
            # capped linear, needs shape > 1 to make sense...needs shape >= 1-r0 else
            # problems at 1
            sl = self.shape
            self.has_mass = (self.r0 > 0)
            self.mass = self.r0

            def g(x):
                return np.where(x == 0, 0, np.minimum(1, self.r0 + sl * x))

            def g_inv(x):
                return np.where(x <= self.r0, 0, (x - self.r0) / sl)

        elif self._name == 'roe' or self._name == 'ccoc':
            # constant roe = capped linear with shape = 1/(1+r), r0=r/(1+r)
            # r = target roe
            self._name = 'ccoc'
            r = self.shape
            v = 1 / (1 + r)
            d = 1 - v
            self.has_mass = (d > 0)
            self.mass = d

            def g(x):
                return np.where(x == 0, 0, np.minimum(1, d + v * x))

            def g_inv(x):
                return np.where(x <= d, 0, (x - d) / v)

            def g_prime(x):
                return v

        elif self._name == 'lep':
            # leverage equivalent pricing
            # self.r0 = risk free/financing and r = risk charge (the solved parameter)
            r = self.shape
            delta = r / (1 + r)
            d = self.r0 / (1 + self.r0)
            spread = delta - d
            self.has_mass = (d > 0)
            self.mass = d

            def g(x):
                return np.where(x == 0, 0, np.minimum(1, d + (1 - d) * x + spread * np.sqrt(x * (1 - x))))

            spread2 = spread ** 2
            a = (1 - d) ** 2 + spread2

            def g_inv(y):
                mb = (2 * (y - d) * (1 - d) + spread2)  # mb = -b
                c = (y - d) ** 2
                rad = np.sqrt(mb * mb - 4 * a * c)
                # l = (mb + rad)/(2 * a)
                u = (mb - rad) / (2 * a)
                return np.where(y < d, 0, np.maximum(0, u))

        elif self._name == 'dual':
            # dual moment
            p = self.shape
            q = 1 / p
            self.has_mass = False

            def g(x):
                return 1 - (1 - x)**p

            def g_inv(y):
                return 1 - (1 - y)**q

            def g_prime(x):
                return p * (1 - x)**(p - 1)

        elif self._name == 'power':
            # power distortion
            # shape = alpha, df = [x0, x1]
            # created from part of power function distribution
            # compare Bernegger approach
            # allows to create distortions with controlled slopes at 0 and 1.
            alpha = float(self.shape)  # numpy complains about powers of integers
            x0, x1 = self.df
            assert x0 < x1, 'Error: x0 must be less than x1'
            self.has_mass = False
            self.mass = 0
            if alpha != 1:
                bl = np.power(x1, -alpha + 1)
                br = np.power(x0, -alpha + 1)
                def g(s):
                    """
                    f(x) = x^-alpha, F(x)=int_{x_0}^x f
                    g(s) = F(x0 + s(x1-x0)) / F(x1)

                    alpha \ge 0 required
                    x0, x1 are other parameters to determine slopes at 0, 1

                    """
                    tl = np.power(x0 + s * (x1 - x0), -alpha + 1)
                    return (tl - br) / (bl - br)

                def g_prime(s):
                    """
                    Derivative of g
                    """
                    tl = np.power(x0 + s * (x1 - x0), -alpha)
                    return (1 - alpha) * (x1 - x0) * tl / (bl - br)

                def g_inv(s):
                    """
                    Inverse of g
                    """
                    t1 = np.power(s * (bl - br) + br, 1 / (1 - alpha))
                    return (t1 - x0) / (x1 - x0)
            else:
                bl = np.log(x1)
                br = np.log(x0)
                def g(s):
                    t = np.log(x0 + s * (x1 - x0))
                    return (t - br) / (bl - br)

                def g_prime(s):
                    return (x1 - x0) / (bl - br) / (x0 + s * (x1 - x0))

                def g_inv(s):
                    return (np.exp(s * (bl - br) + br) - x0) / (x1 - x0)

        elif self._name == 'bitvar':
            # bitvar tvar, df = p0 <p1, shape = weight on p1
            try:
                p0, p1 = self.df
                assert p0 < p1, 'Error: p0 must be less than p1'
                w = self.shape
                self.has_mass = (p1 == 1)
                self.mass = w if p1 == 1 else 0
                pt = (1 - p1) / (1 - p0) * (1 - w) + w
                # if p1==1 there is a mass, so have to be careful to put that in
                if p1 == 1:
                    # if p1 = 1 the result is a TVaR + max
                    alpha = 1 / (1 - p0)
                    def g(x):
                        # <= 0 handles rounding issues gracefully
                        return w * np.where(x<=0, 0, 1) + (1 - w) * np.minimum(alpha * x, 1)

                    # this stupid code with fixed number wasted two days of my life.
                    # but we don't really use g_inv so...
                    s = np.array([0.,   1e-50, 1-p0, 1.])
                    gs = np.array([0.,     pt, 1.,   1.])
                    g_inv = interp1d(gs, s, kind='linear', bounds_error=False, fill_value=(0,1))
                else:
                    s = np.array([0.,  1 - p1, 1-p0, 1.])
                    gs = np.array([0.,     pt, 1.,   1.])
                    g = interp1d(s, gs, kind='linear', bounds_error=False, fill_value=(0,1))
                    g_inv = interp1d(gs, s, kind='linear', bounds_error=False, fill_value=(0,1))

                if p1 < 1:
                    def g_prime(x):
                        return np.where(x > 1 - p0, 0,
                                        np.where(x < 1 - p1,
                                                 w / (1 - p1) + (1 - w) / (1 - p0),
                                                 (1 - w) / (1 - p0)))
                else:
                    # p1==1, don't want 1/(1-p1)
                    def g_prime(x):
                        return np.where(x > 1 - p0, 0, (1 - w) / (1 - p0))

            except:
                raise ValueError('Inadmissible parameters to Distortion for bitvar. '
                                 'Pass shape=wt for p1 and df=[p0, p1]')

        elif self._name == 'wtdtvar':
            # Create the weighted tvar g and g_inv.
            # ps = p values of TVaRs passed as shape
            # wts = weights, must sum to 1, passed as df

            # extract inputs
            ps = np.array(self.shape)
            wts = np.array(self.df)
            if 1 in ps and wts[-1] > 0:
                self.has_mass = True
                self.mass = wts[-1]
            else:
                self.has_mass = False
                self.mass = 0

            if self.display_name == '':
                self.display_name = f'wtdTVaR on {len(ps):d} points'

            # data checks
            # check sorted
            assert np.all(ps[:-1] < ps[1:]), 'Error: ps must be sorted ascending.'

            # ok, sum to 1 is a user requirement, this is too painful!
            # swts = wts.sum()
            # assert (swts==1
            #         or np.nextafter(swts, np.inf)==1
            #         or np.nextafter(swts, -np.inf)==1
            #        ), f'Error: weights must sum to 1. Entered weights sum to {wts.sum()}'

            # must ensure 0 and 1 are in ps; add with zero weights
            # if missing. This ensures resulting function has
            # g(0)=0 and g(1)=1
            if 0 not in ps:
                ps = np.insert(ps, 0, 0)
                wts = np.insert(wts, 0, 0)
            if 1 not in ps:
                ps = np.append(ps, 1)
                wts = np.append(wts, 0)
            else:
                # if input with a mass at 1 need to adjust
                # so the interpolation "sees" it
                ps[-1] = np.nextafter(1, -1)
                ps = np.append(ps, 1)
                wts = np.append(wts, 0)

            # evaluate at knot points and weighted tvar at knot points
            s = 1 - ps[::-1]
            gs = wts @ Distortion.tvar_terms(ps)
            g = interp1d(s, gs, kind='linear', bounds_error=False, fill_value=(0,1))
            g_inv = interp1d(gs, s, kind='linear', bounds_error=False, fill_value=(0,1))

        elif self._name  == 'minimum':
            # min of several distortion; shape is list of Distortions
            self.has_mass = np.all([d.has_mass for d in self.shape])
            if self.has_mass:
                self.mass = np.min([d.mass for d in self.shape])
            else:
                self.mass = 0

            if self.display_name == '':
                self.display_name = f'Minimum of {len(self.shape):d} distortions'

            loc_shape = self.shape  # #.copy() ## ? insulates from shape?
            def min_index(x):
                g_values = np.array([gi.g(x) for gi in loc_shape])
                return np.argmin(g_values, axis=0)
            self.min_index = min_index

            def g(x):
                g_values = np.array([gi.g(x) for gi in loc_shape])
                return np.min(g_values, axis=0)

            def g_prime(x):
                """
                Note: you really need to check whether the point x is a
                knot point for the minimum function.
                """
                # need to see what happens just before 1.
                # This is a hack.
                x = np.where(x==1, 1 - 1e-15, x)
                g_values = np.array([gi.g(x) for gi in loc_shape])
                # Find the index of the minimum value
                min_index = np.argmin(g_values, axis=0)
                # Return the derivative (g_prime) of the object that achieved the minimum
                if np.isscalar(min_index):
                    return loc_shape[min_index].g_prime(x)
                elif np.isscalar(x):
                    return np.array([loc_shape[i].g_prime(x) for i in min_index])
                else:
                    return np.array([loc_shape[i].g_prime(xi) for i, xi in zip(min_index, x)])

            def g_inv(y):
                """
                max inverse value is inverse of min: draw a picture to see why
                """
                inv_values = np.array([gi.g_inv(y) for gi in loc_shape])
                # Return the maximum of these inverse values
                return np.max(inv_values, axis=0)

        elif self._name == 'mixture':
            # mixture of several distortion; shape is list of Distortions
            # and weights are passed in df
            # reconstuct if needed (pickle)
            loc_shape = []
            for x in self.shape:
                if isinstance(x, Distortion):
                    loc_shape.append(x)
                else:
                    loc_shape.append(Distortion(**x))
            self.has_mass = np.any([d.has_mass for d in loc_shape])
            if self.has_mass:
                self.mass = np.sum([d.mass * w for (d, w) in zip(loc_shape, self.df)])
            else:
                self.mass = 0

            if self.display_name == '':
                self.display_name = f'Mixture of {len(loc_shape):d} distortions'

            if self.df is None:
                loc_wts = self.df = np.array([1 / len(loc_shape)] * len(loc_shape))
            else:
                loc_wts = np.array(self.df.copy())
            def g(x):
                g_values = np.array([gi.g(x) for gi in loc_shape])
                # at this point, have to be a bit careful about the shape of the arrays
                if g_values.ndim > 2:
                    # Flatten the higher dimensions (n1, n2) into a single dimension
                    vals_flat = g_values.reshape(len(loc_shape), -1)  # shape (2, n1 * n2)

                    # Perform the matrix multiplication
                    result_flat = loc_wts @ vals_flat  # shape (n1 * n2)

                    # Reshape the result back to the original shape (n1, n2)
                    result = result_flat.reshape(g_values.shape[1], g_values.shape[2])
                else:
                    # vals is already 2D, just apply the @ operation directly
                    result = loc_wts @ g_values

                return result

            def g_prime(x):
                g_values = np.array([gi.g_prime(x) for gi in loc_shape])
                # at this point, have to be a bit careful about the shape of the arrays
                if g_values.ndim > 2:
                    # Flatten the higher dimensions (n1, n2) into a single dimension
                    vals_flat = g_values.reshape(len(loc_shape), -1)
                    result_flat = loc_wts @ vals_flat  # shape (n1 * n2)
                    result = result_flat.reshape(g_values.shape[1], g_values.shape[2])
                else:
                    # vals is already 2D, just apply the @ operation directly
                    result = loc_wts @ g_values

                return result

            def g_inv(y):
                """
                Inverse of mixture...֫
                """
                raise NotImplementedError('Inverse of mixture not implemented')

        elif self._name == 'convex':
            # convex envelope and general interpolation
            # NOT ALLOWED to have a mass...why not???
            # evil - use shape to indicate if mass at zero
            if self.shape > 0:
                self.has_mass = True
                self.mass = self.shape

            # use shape for number of points in calibrating data set
            if self.display_name == '':
                self.display_name = f'Convex on {len(self.df):d} points'
            if not (0 in self.df[self.col_x].values and 1 in self.df[self.col_x].values):
                # painful...always want 0 and 1 there...but don't know what other columns in df
                # logger.debug('df does not contain s=0/1...adding')
                self.df = self.df[[self.col_x, self.col_y]].copy().reset_index(drop=True)
                self.df.loc[len(self.df)] = (0,0)
                self.df.loc[len(self.df)] = (1,1)
                self.df = self.df.sort_values(self.col_x)
            if len(self.df) > 2:
                hull = ConvexHull(self.df[[self.col_x, self.col_y]])
                knots = list(set(hull.simplices.flatten()))
                g = interp1d(self.df.iloc[knots, self.df.columns.get_loc(self.col_x)],
                             self.df.iloc[knots, self.df.columns.get_loc(self.col_y)], kind='linear',
                             bounds_error=False, fill_value=(0,1))
                g_inv = interp1d(self.df.iloc[knots, self.df.columns.get_loc(self.col_y)],
                             self.df.iloc[knots, self.df.columns.get_loc(self.col_x)], kind='linear',
                             bounds_error=False, fill_value=(0,1))
            else:
                self.df = self.df.sort_values(self.col_x)
                g = interp1d(self.df[self.col_x], self.df[self.col_y], kind='linear',
                             bounds_error=False, fill_value=(0,1))
                g_inv = interp1d(self.df[self.col_y], self.df[self.col_x], kind='linear',
                             bounds_error=False, fill_value=(0,1))
        else:
            print(self._name, self.shape, self.df)
            raise ValueError(
                f"Incorrect spec passed to Distortion; name={self._name}, shape={self.shape}, r0={self.r0}, df={self.df}")

        self.g = g
        self.g_inv = g_inv
        if g_prime is None:
            g_prime = lambda x: (g(x + 1e-6) - g(x - 1e-6)) / 2e-6
        self.g_prime = g_prime

    # utility methods to create usual suspects and other common distortions
    @staticmethod
    def tvar(p):
        """
        Utility method to create tvar.
        """
        return Distortion('tvar', p, display_name=f'TVaR({p:.3g})')

    @staticmethod
    def max():
        """
        Utility method to create max = TVaR 1.
        """
        p = 1.
        return Distortion('tvar', p, display_name='max')

    @staticmethod
    def mean():
        """
        Utility method to create mean = TVaR 0.
        """
        p = 0.
        return Distortion('tvar', p, display_name='mean')

    @staticmethod
    def wang(shape):
        """
        Utility method to create wang.
        """
        return Distortion('wang', shape, display_name=f'Wang({shape:.3g})')

    @staticmethod
    def ph(shape):
        """
        Utility method to create ph.
        """
        return Distortion('ph', shape, display_name=f'PH({shape:.3g})')

    @staticmethod
    def dual(shape):
        """
        Utility method to create dual.
        """
        return Distortion('dual', shape, display_name=f'dual({shape:.3g})')

    @staticmethod
    def bitvar(p0, p1, w=0.5):
        """
        Utility method to create bitvar with :math:`p_0 < p_1`. The
        weight w is the weight on p1. Remember p=0 corresponds to
        the mean and p=1 to the max.
        """
        # first check if it is really a TVaR
        if p0 == p1:
            return Distortion.tvar(p0)
        elif w == 0:
            return Distortion.tvar(p0)
        elif w == 1:
            return Distortion.tvar(p1)
        else:
            return Distortion('bitvar', w, df=[p0, p1], display_name=f'bitvar({p0:.3g}, {p1:.3g}; {w:.3g})')

    @staticmethod
    def ccoc(d):
        """
        Utility method to create ccoc with given discount factor. The
        default constructor inputs the return r instead of the discount factor d.
        d = r / (1 + r).
        """
        r = d / (1. - d)
        return Distortion('ccoc', r, display_name=f'ccoc({r:.3g})')

    @staticmethod
    def minimum(distortion_list):
        """
        Utility method to create minimum of a list of distortions.
        """
        return Distortion('minimum', distortion_list,
                          display_name=f'minimum({len(distortion_list)})')

    @staticmethod
    def mixture(distortion_list, weights=None):
        """
        Utility method to create mixture of a list of distortions.
        """
        return Distortion('mixture', distortion_list, df=weights,
                          display_name=f'mixture({len(distortion_list)})')

    @staticmethod
    def beta(a, b):
        """
        Utility method to create mixture of a list of distortions.
        """
        return Distortion('beta', [a, b], display_name=f'beta({a:.3f}, {b:.3f})')

    @staticmethod
    def power(alpha, x0, x1):
        """
        Utility method to create mixture of a list of distortions.
        """
        return Distortion('power', alpha, df=[x0, x1],
                          display_name=f'power({alpha:.3f}, {x0:.3f}, {x1:.3f})')

    def g_dual(self, x):
        """
        The dual of the distortion function g.
        """
        return 1 - self.g(1 - x)

    def id(self):
        """
        Unique ID as a short string
        """
        # operational end of the hose
        bit = {k: v for k, v in self.__dict__.items() if k in ('_name', 'r0', 'df', 'shape', 'col_x', 'col_y')}
        return short_hash(str(bit))

    def __str__(self):
        return self.name
        # if self.display_name != '':
        #     s = self.display_name
        #     return s
        # else:
        #     return self.__repr__()
        # was
        # elif isinstance(self.shape, (list, tuple, str)):
        #     s = f'{self._distortion_names_.get(self._name, self._name)}, {self.shape}'
        # else:
        #     s = f'{self._distortion_names_.get(self._name, self._name)}, {self.shape:.3f}'
        # if self._name == 'tt':
        #     s += f', {self.df:.2f}'
        # if self._name == 'wtdtvar':
        #     s += f', ({self.df[0]:.3f}/{self.df[1]:.3f})'
        # elif self.has_mass:
        #     # don't show the mass for wtdtvar
        #     s += f', mass {self.mass:.3f}'
        # return s

    def __repr__(self):
        return self.name
        # Get the default __repr__ output
        # this messes up plot etc. that uses repr as default title
        # default_repr = super().__repr__()
        # return f'{self.name}: {default_repr}'
        # # originally
        # if self.has_mass:
        #     s += f', {self.r0})'
        # elif self._name == 'tt':
        #     s += f', {self.df:.2f})'
        # else:
        #     s += ')'
        # return s

    @property
    def name(self):
        return self.display_name if self.display_name != '' else self._name

    @name.setter
    def name(self, value):
        self._name = value

    def plot(self, xs=None, n=101, both=True, ax=None, plot_points=True, scale='linear',
             c=None, c_dual=None, size='small', **kwargs):
        """
        Quick plot of the distortion

        :param xs:
        :param n:  length of vector is no xs
        :param both: True: plot g and g_dual and add decorations, if False just g and no trimmings.
        :param ax:
        :param plot_points:
        :param scale: linear as usual or return plots -log(gs)  vs -logs and inverts both scales
        :param size: 'small' or 'large' for size of plot, FIG_H or FIG_W. The default is 'small'.
        :param kwargs:  passed to matplotlib.plot
        :return:
        """

        assert scale in ['linear', 'return']

        if scale == 'return' and n == 101:
            # default not enough for return
            n = 10001

        if xs is None:
            xs = np.hstack((0, np.linspace(1e-10, 1, n)))

        y1 = self.g(xs)
        y2 = None   # keep linter happy
        if both:
            y2 = self.g_dual(xs)

        if ax is None:
            if size == 'small':
                sz = FIG_H
            elif isinstance(size, (float, int)):
                sz = size
            else:
                sz = FIG_W
            fig, ax = plt.subplots(1,1, figsize=(sz, sz), constrained_layout=True)

        if c is None:
            c = 'C1'
        if c_dual is None:
            c_dual = 'C2'
        if scale == 'linear':
            ax.plot(xs, y1, c=c, label=self.name, **kwargs)
            if both:
                ax.plot(xs, y2, c=c_dual, label='$g\check$', **kwargs)
            ax.plot(xs, xs, color='k', lw=0.5, alpha=0.5)
            # ax.plot(xs, xs, lw=0.5, color='C0', alpha=0.5)
        elif scale == 'return':
            ax.plot(xs, y1, c=c, label='$g$', **kwargs)
            if both:
                ax.plot(xs, y2, c=c_dual, label='$g\check$', **kwargs)
            ax.set(xscale='log', yscale='log', xlim=[1/5000, 1], ylim=[1/5000, 1])
            ax.plot(xs, xs, color='k', lw=0.5, alpha=0.5)

        if self._name == 'convex' and plot_points:
            if len(self.df) > 50:
                alpha = .35
            elif len(self.df) > 20:
                alpha = 0.6
            else:
                alpha = 1
            if c is None:
                c = 'C4'
            if scale == 'linear':
                ax.scatter(x=self.df[self.col_x], y=self.df[self.col_y], marker='.', s=15, color=c, alpha=alpha)
            elif scale == 'return':
                ax.scatter(x=1/self.df[self.col_x], y=1/self.df[self.col_y], marker='.', s=15, color=c, alpha=alpha)

        ax.set(title=self.name, aspect='equal',
               xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
        if both:
            ax.legend(loc='upper left', fontsize='x-small')
        return ax

    @classmethod
    def test(cls, r0=0.035, df=[0.0, .9]):  # noqa default class is mutable
        """
        Tester: make some nice plots of available distortions.

        :return:
        """

        f0, axs0 = plt.subplots(2, 11, figsize=(22, 4), constrained_layout=True, sharex=True, sharey=True)
        f1, axs1 = plt.subplots(2, 11, figsize=(22, 4), constrained_layout=True, sharex=True, sharey=True)
        axiter0 = iter(axs0.flat)
        axiter1 = iter(axs1.flat)

        xs = np.linspace(0, 1, 1001)

        # zip stops at the shorter of the vectors, so this does not include convex (must be listed last)
        # added df for the t; everyone else can ignore it
        # rank by order on large lsoses...
        for axiter, scale in zip([axiter0, axiter1], ['linear', 'return']):
            for name, shape in zip(cls._available_distortions_, cls._eg_param_1_):
                dist = Distortion(name, shape, r0, df=df)
                dist.plot(xs, ax=next(axiter), scale=scale)

            dist = Distortion.convex_example('bond')
            dist.plot(xs, ax=next(axiter), scale=scale)

            # order will look better like this
            for name, shape in zip(cls._available_distortions_, cls._eg_param_2_):
                dist = Distortion(name, shape, r0, df=df)
                dist.plot(xs, ax=next(axiter), scale=scale)

            dist = Distortion.convex_example('cat')
            dist.plot(xs, ax=next(axiter), scale=scale)

        # tidy up
        for ax in axiter0:
            f0.delaxes(ax)
        for ax in axiter1:
            f1.delaxes(ax)

        f0.suptitle('Example Distortion Functions - Linear Scale')
        f1.suptitle('Example Distortion Functions - Return Scale')

    @staticmethod
    def distortions_from_params(params, index, r0=0.025, df=5.5, pricing=True, strict=True):
        """
        Make set of dist funs and inverses from params, output of port.calibrate_distortions.
        params must just have one row for each method and be in the output format of cal_dist.

        Called by Portfolio.

        :param index:
        :param params: dataframe such that params[index, :] has a [lep, param] etc.
               pricing=True, strict=True: which distortions to allow
               df for t distribution
        :param r0: min rol parameters
        :param strict:
        :param pricing:
        :return:
        """

        temp = params.loc[index, :]
        dists = {}
        for dn in Distortion.available_distortions(pricing=pricing, strict=strict):
            param = float(temp.loc[dn, 'param'])
            dists[dn] = Distortion(name=dn, shape=param, r0=r0, df=df)

        return dists

    @staticmethod
    def convex_example(source='bond'):
        """
        Example convex distortion using data from https://www.bis.org/publ/qtrpdf/r_qt0312e.pdf.

        :param source: bond gives a bond yield curve example, cat gives cat bond / cat reinsurance pricing based example
        :return:
        """

        if source == 'bond':
            yield_curve = '''
            AAAA    0.000000  0.000000
            AAA     0.000018  0.006386
            AA      0.000144  0.007122
            A       0.000278  0.010291
            BBB     0.002012  0.017089
            BB      0.012674  0.036455
            B       0.040052  0.069181
            Z       1.000000  1.000000'''

            df = pd.read_fwf(StringIO(yield_curve))
            df.columns = ['Rating', 'EL', 'Yield']
            return Distortion('convex', 'Yield Curve', df=df, col_x='EL', col_y='Yield')

        elif source.lower() == 'cat':
            cat_bond = '''EL,ROL
            0.116196,0.32613
            0.088113,0.2452
            0.074811,0.22769
            0.056385,0.17131
            0.046923,0.15326
            0.032961,0.12222
            0.02807,0.11037
            0.024205,0.1022
            0.011564,0.07284
            0.005813,0.06004
            0,0
            1,1'''
            df = pd.read_csv(StringIO(cat_bond))
            return Distortion('convex', 'Cat Bond', df=df, col_x='EL', col_y='ROL')

        else:
            raise ValueError(f'Inadmissible value {source} passed to convex_example, expected yield or cat')

    @staticmethod
    def bagged_distortion(data, proportion, samples, display_name=""):
        """
        Make a distortion by bootstrap aggregation (Bagging) resampling, taking the convex envelope,
        and averaging from data.

        Each sample uses proportion of the data.

        Data must have  two columns: EL and Spread

        :param data:
        :param proportion: proportion of data for each sample
        :param samples: number of resamples
        :param display_name: display_name of created distortion
        :return:
        """

        df = pd.DataFrame(index=np.linspace(0,1,10001), dtype=float)

        for i in range(samples):
            rebit = data.sample(frac=proportion, replace=False, random_state=RANDOM)
            rebit.loc[-1] = [0, 0]
            rebit.loc[max(rebit.index)+1] = [1, 1]
            d = Distortion('convex', 0, df=rebit, col_x='EL', col_y='Spread')
            df[i] = d.g(df.index)

        df['avg'] = df.mean(axis=1)
        df2 =df['avg'].copy()
        df2.index.name = 's'
        df2 = df2.reset_index(drop=False)

        d = Distortion('convex', 0, df=df2, col_x='s', col_y='avg', display_name=display_name)

        return d

    @staticmethod
    def average_distortion(data, display_name, n=201, el_col='EL', spread_col='Spread'):
        """
        Create average distortion from (s, g(s)) pairs. Each point defines a wtdTVaR with
        p=s and p=1 points.

        :param data:
        :param display_name:
        :param n: number of s values (between 0 and max(EL), 1 is added
        :param el_col:   column containing EL
        :param spread_col: column containing Spread
        :return:
        """

        els = data[el_col]
        spreads = data[spread_col]
        max_el = els.max()
        s = np.hstack((np.linspace(0, max_el, n), 1))
        ans = np.zeros((len(s), len(data)))
        for i, el, spread in zip(range(len(data)), els, spreads):
            p = 1 - el
            w = (spread - el) / (1 - el)
            d = Distortion('wtdtvar', w, df=[0,p])
            ans[:, i] = d.g(s)

        df = pd.DataFrame({'s': s, 'gs': np.mean(ans, 1)})
        dout = Distortion('convex', None, df=df, col_x='s', col_y='gs', display_name=display_name)
        return dout

    # replacedc with wtdtvar type Distortions
    # @staticmethod
    # def wtd_tvar(ps, wts, display_name='', details=False):
    #     """
    #     A careful version of wtd tvar with knots at ps and wts.
    #
    #     :param ps:
    #     :param wts:
    #     :param display_name:
    #     :param details:
    #     :return:
    #     """
    #
    #     # evaluate at 0, 1 and all the knot points
    #     ps0 = np.array(ps)
    #     s = np.array(sorted(set((0.,1.)).union(1-ps0)))
    #     s = s.reshape(len(s), 1)
    #
    #     wts = np.array(wts).reshape(len(wts), 1)
    #     if np.sum(wts) != 1:
    #         wts = wts / np.sum(wts)
    #     ps = np.array(ps).reshape(1, len(ps))
    #
    #     gs = np.where(ps == 1, 1, np.minimum(s / (1 - ps), 1)) @ wts
    #
    #     d = Distortion.s_gs_distortion(s, gs, display_name)
    #     if details:
    #         return d, s, gs
    #     else:
    #         return d

    @staticmethod
    def s_gs_distortion(s, gs, display_name=''):
        """
        Make a convex envelope distortion from {s, g(s)} points.
        TODO: allow mass at zero; currently shape=0 passes to no mass at zero
        even if s,gs implies one. LAZY
        :param s: iterable (can be converted into numpy.array
        :param gs:
        :param display_name:
        :return:
        """
        s = np.array(s)
        gs = np.array(gs)
        return Distortion('convex', 0, df=pd.DataFrame({'s': s.flat, 'gs': gs.flat}),
                          col_x='s', col_y='gs', display_name=display_name)

    @staticmethod
    def random_distortion(n_knots, mass=0, mean=0, wt_rng=None, random_state=None):
        """
        Create a random distortion. if mass (mean)
        add a mass (mean) term.
        wt_rng to generate (spiky) weights, eg. wt_rng=ss.pareto(1.5).rvs.
        """
        ps = np.random.rand(n_knots)
        ps.sort()
        if wt_rng is None:
            wts = np.random.rand(n_knots)
        else:
            wts = wt_rng(size=n_knots, random_state=random_state)
        wts = wts / wts.sum(dtype=np.float64) * (1 - mass - mean)
        mn = ''
        ma = ''
        if mass:
            ps = np.append(ps, 1)
            wts = np.append(wts, mass)
            ma = f', mx={mass:.3f}'
        if mean:
            ps = np.insert(ps, 0, 0)
            wts = np.insert(wts, 0, mean)
            mn = f', mn={mean:.3f}'
        return Distortion('wtdtvar', ps, df=wts, display_name=f'Rnd {n_knots} knots{mn}{ma}')

    def price(self, ser, a=np.inf, kind='ask', S_calculation='forwards'):
        r"""
        Compute the bid and ask prices for the distribution determined by ``ser`` with
        an asset limit ``a``. Index of ``ser`` need not be equally spaced, so it can
        be applied to :math:`\kappa`. However, the index values must be distinct.
        Index is sorted if needed.

        To apply to do kappa for unit A in portfolio port::

            ser = port.density_df[['exeqa_A', 'p_total']].\
                set_index('exeqa_A').groupby('exeqa_A').\
                sum()['p_total']
            dist.price(ser, port.q(0.99), 'both')

        Always use ``S_calculation='forwards`` method to compute S = 1 - cumsum(probs).
        Computes the price as the integral of gS.

        Returns a tuple of (premium, loss) for the ask or bid price. When 'both' requested
        returns (bid_premium, loss, ask_premium).

        Generally prefer to use newer price_ex.
        Requires ser starts at x=0.

        :param ser: pd.Series of is probabilities, indexed by outcomes. Outcomes need not
          be spaced evenly. ``ser`` is usually a probability column from ``density_df``.
        :param kind: is "ask", "bid",  or "both", giving the pricing view.
        :param a: asset level. ``ser`` is truncated at ``a``.
        """

        if not isinstance(ser, pd.Series):
            raise ValueError(f'ser must be a pandas Series, not {type(ser)}')

        if kind in ['bid', 'ask']:
            pass
        elif kind == 'both':
            return [*self.price(ser, a, 'bid', S_calculation),
                    self.price(ser, a, 'ask', S_calculation)[0]]
        else:
            raise ValueError(f'kind must be bid, ask, or both, not {kind}')

        # unlikely to be an issue
        if not ser.index.is_monotonic_increasing:
            ser = ser.sort_index(ascending=True)

        # apply limit
        if a < np.inf:
            # ser = ser.copy()
            # tail = ser[a:].sum()
            # fixed bug: need loc to include a
            ser = ser.loc[:a]
            # ser[a] = tail

        if S_calculation == 'forwards':
            S = 1 - ser.cumsum()
            S = np.maximum(0, S)
        else:
            fill_value = max(0, 1 - ser.sum())
            S = np.minimum(1, fill_value +
                           ser.shift(-1, fill_value=0)[::-1].cumsum()[::-1])

        # not all distortions return numpy; force conversion
        if kind == 'ask':
            gS = np.array(self.g(S))
        else:
            gS = np.array(self.g_dual(S))

        # trapz is not correct with our interpretation elsewhere; we use a step function
        # loss = np.trapz(S, S.index)
        # prem = np.trapz(gS, S.index)
        dx = np.diff(S.index)
        loss = (S.iloc[:-1].values * dx).sum()
        prem = (gS[:-1] * dx).sum()
        return prem, loss

    def price2(self, ser, a=None, S_calculation='forwards'):
        r"""
        Similar to price, and uses the exact same calculation. Returns bid and ask by
        default in a DataFrame. Returns all cumulative sums if a is None (likely, not
        what you want).

        Generally prefer to use newer price_ex.
        Requires ser starts at x=0.
        """

        if not isinstance(ser, pd.Series):
            raise ValueError(f'ser must be a pandas Series, not {type(ser)}')

        # unlikely to be an issue
        if not ser.index.is_monotonic_increasing:
            ser = ser.sort_index(ascending=True)

        if S_calculation == 'forwards':
            S = 1 - ser.cumsum()
            S = np.maximum(0, S)
        else:
            fill_value = max(0, 1 - ser.sum())
            S = np.minimum(1, fill_value +
                           ser.shift(-1, fill_value=0)[::-1].cumsum()[::-1])

        # not all distortions return numpy; force conversion
        gS = np.array(self.g(S))
        dgS = np.array(self.g_dual(S))

        dx = np.diff(S.index)
        loss = (S.iloc[:-1].values * dx).cumsum()
        ask = (gS[:-1] * dx).cumsum()
        bid = (dgS[:-1] * dx).cumsum()
        # index is shifted by 1 because it is the right hand range of integration
        ans = pd.DataFrame({'bid': bid, 'el': loss, 'ask': ask}, index=S.index[1:])
        if a is None:
            return ans
        else:
            # no longer guaranteed that a is in ser.index
            return ans.iloc[ans.index.get_indexer([a], method='nearest')]

    def price_ex(self, ser, a=np.inf, kind='ask', method='dx', S_calculation='backwards',
                 as_frame=False):
        """
        Updated version of price and price2 that should be used in the future.

        Always uses S_calculation='forwards' method to compute S = 1 - cumsum(probs).
        Sorts ser if needed.

        if calculation 'dx' computes the price as the integral of gS dx. (This method
        has fewer diffs).
        if calculation 'ds' computes the prices as the integralk of x d(gS).

        Both methods require the index to be sorted, so that is checked and handled

        Neither method requires the index to be unique.

        kind = ask, bid or both.

        Asset level a is required to be in the index of ser, there is no interpolation.

        There is an ambiguity here about a. If you want the unlimited price of an
        unbounded variable then you integrate to infinity. In the numerical approximation
        you integrate over all ser values. In that canse there is no a P(X>a) term
        added in the dgS method, because Pr(X>inf)=0. However, if X is actually bounded
        and as a mass at its max value then you do need to add xPr(X>

        Returns a namedtuple.

        summarize=True automatically summarizes the index if it is not unique.

        if as_frame is True, returns a DataFrame with bid, el, and ask columns,
        otherwise returns a namedtuple with bid, el, and ask fields.

        If S_calculation is 'forwards', S is computed as 1 - ser.cumsum(). If 'backwards',
        it is computed as p_total[::-1].cumsum()[::-1].shift(-1, fill_value=0). This is computed
        before p is truncated for a.

        Because of potential underflow issues, the backwards method for computing the ask
        is more reliable. Therefore it has become the default.

        When there are thin tails, neither method is reliable for the bid
        because g_dual(s) = 1 - g(1 - s).
        """
        # input audits
        assert kind in ['bid', 'ask', 'both'], "kind must be 'bid', 'ask', or 'both'"
        assert method in ['dx', 'ds'], "method must be 'dx' or 'ds'"
        assert S_calculation in ['forwards', 'backwards'], "S_calculation must be 'forwards' or 'backwards'"
        if not isinstance(ser, pd.Series):
            raise ValueError(f'ser must be a pandas Series, not {type(ser)}')
        assert ser.index.is_monotonic_increasing, 'ser index must be sorted ascending'
        # need
        if not ser.index.is_monotonic_increasing:
            ser = ser.sort_index(ascending=True)

        if S_calculation == 'forwards':
            # truncate: note here you need ser a Series and you need to use
            # .loc to truncate at a and include a. Otherwise, it may not be included.
            if a < np.inf:
                assert a in ser.index, f'a={a} must be in the index of ser'
                # notice this includes a at the end, which is not needed
                # to compute the integral Sdx integral (but is for the xdS)
                ser = ser.loc[:a]
            # calculation using forwards method
            S = np.maximum(0, 1 - ser.cumsum())
            if a == np.inf:
                # if a = inf then you want the last prob to be zero come what may
                S.iloc[-1] = 0
        elif S_calculation == 'backwards':
            # there is an implicit assumption here that ser.sum() == 1.
            if not np.allclose((ser.sum()), (1)):
                print(f'WARNING: ser.sum() = {ser.sum()} is not 1.')
            S = ser[::-1].cumsum().shift(1, fill_value=0)[::-1]
            S = np.minimum(1, S)
            if a < np.inf:
                assert a in ser.index, f'a={a} must be in the index of ser'
                S = S.loc[:a]
            # note that by construction the last element here is automatically zero

        gS = dual_gS = None
        el = bid = ask = np.nan   # keep the linter happy

        if kind == 'ask':
            gS = np.array(self.g(S))
        elif kind == 'both':
            gS = np.array(self.g(S))
            dual_gS = np.array(self.g_dual(S))
        elif kind == 'bid':
            dual_gS = np.array(self.g_dual(S))

        # at this point S is a Series, and gS and dual_gS are numpy arrays
        if method == 'dx':
            # note: the last value of S is for prob > a, so it is not used.
            dx = np.diff(S.index)
            x0 = S.index[0]
            el = (S.iloc[:-1].values * dx).sum() + x0
            if gS is not None:
                ask = (gS[:-1] * dx).sum() + x0
            if dual_gS is not None:
                bid = (dual_gS[:-1] * dx).sum() + x0

        elif method == 'ds':
            # the last adjustment to the xdgS integral is to add
            # a P(X>a) provided a is finite. To simplify math:
            if a == np.inf:
                a = 0
            dS = np.diff(S, prepend=1.)
            x = np.array(S.index)
            # S is a Series
            # int gS dx = - x S + a S - 0 S0
            el = -((x * dS).sum()) + a * S.iloc[-1]
            if gS is not None:
                dgS = np.diff(gS, prepend=1.)
                ask = -((x * dgS).sum()) +a * gS[-1]
            if dual_gS is not None:
                ddual_gS = np.diff(dual_gS, prepend=1.)
                bid = -((x * ddual_gS).sum()) + a * dual_gS[-1]

        if as_frame:
            return pd.DataFrame([[bid, el, ask, self.name, a]],
                                index=[0],  # convenient to know it is 0
                                # index=pd.MultiIndex.from_arrays([[self.name], [a]],
                                #                                 names=['distortion', 'assets']),
                                columns=pd.Index(
                                    ['bid', 'el', 'ask', 'distortion', 'assets']))
        else:
            # Define a namedtuple called 'Point' with fields 'x' and 'y'
            Price = namedtuple('Price', 'bid,el,ask')
            return Price(bid, el, ask)

    # add the numba functions to the class for easy access
    def quick_gS(self, den):
        """
        Calls numba version to compute ``self.g(1-den.sumsum())``
        for tvar or bitvar distortions.
        """
        if self._name == 'tvar':
            p = self.shape
            if isinstance(den, pd.Series):
                return tvar_gS(den.values, p)
            else:
                return tvar_gS(den, p)
        elif self._name == 'bitvar':
            p0, p1 = self.df
            w = self.shape
            if isinstance(den, pd.Series):
                return bitvar_gS(den.values, p0, p1, w)
            else:
                return bitvar_gS(den, p0, p1, w)
        else:
            raise ValueError(f'quick_gS only implemented for TVaR and BiTVaR, not {self._name}')

    def quick_ra(self, den, x=None):
        """
        Computes the risk adjusted expected value of ``den`` and ``x`` using the distortion.
        Calls numba version of ra for tvar or bitvar distortion. If ``x is None``,
        requires ``den`` to be a ``Series`` and uses the index. ``x`` values must be in
        ascending order.
        """
        if self._name == 'tvar':
            p = self.shape
            if isinstance(den, pd.Series):
                return tvar_ra(den.values, np.array(den.index), p)
            else:
                assert x is not None
                return tvar_ra(den, x, p)
        elif self._name == 'bitvar':
            p0, p1 = self.df
            w = self.shape
            if isinstance(den, pd.Series):
                return bitvar_ra(den.values, np.array(den.index), p0, p1, w)
            else:
                return bitvar_ra(den, x, p0, p1, w)
        else:
            raise ValueError(f'quick_ra only implemented for TVaR and BiTVaR, not {self._name}')


def approx_ccoc(roe, eps=1e-14, display_name=None):
    """
    Create a continuous approximation to the CCoC distortion with return roe.
    Helpful utility function for creating a distortion.

    :param roe: return on equity
    :param eps: small number to avoid mass at zero
    """

    return Distortion('bitvar', roe/(1 + roe), df=[0, 1-eps],
                      display_name=f'aCCoC {roe:.2%}' if display_name is None
                      else display_name
                      )


def tvar_weights(d):
    """
    Return tvar weight function for a distortion d. Use np.gradient to differentiate g' but
    adjust for certain distortions. The returned function expects a numpy array of p
    values.

    :param: d distortion
    """

    shape = d.shape
    r0 = d.r0
    nm = d.name

    if nm.lower().find('ccoc') >= 0:
        nm = 'ccoc'
        v = shape
        def wf(p):
            return np.where(p==0, 1 - v,
                            # use nan to try and distinguish mass from density
                            np.where(p == 1, v, np.nan))
    elif nm == 'ph':
        # this is easy, do by hand
        def wf(p):
            return np.where(
                p==1, shape,  # really a mass!
                -shape * (shape - 1) * (1 - p) ** (shape - 1)
            )
    elif nm == 'tvar':
        def wf(p):
            # something that will plot reasonably
            dp = p[1] - p[0]
            return np.where(np.abs(p - shape) < dp, 1, 0)

    else:
        # numerical approximation
        def wf(p):
            gprime = d.g_prime(1 - p)
            wt = (1 - p) * np.gradient(gprime, p)
            return wt

        # adjust for endpoints in certain situations (where is a mass at 0 or a kink
        # s=1 (slope approaching s=1 is > 0)

        if nm == 'wang':
            pass
        elif nm == 'dual':
            pass

    return wf  #noqa