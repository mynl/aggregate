import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from io import StringIO
import pandas as pd
from textwrap import fill
import logging

from .constants import *
from .random_agg import RANDOM

logger = logging.getLogger(__name__)


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
    _eg_param_1_ =              (.9,     .1,      .9,    0.25,  0.8,   1.1,   1.5,   .1,     0.15,     .15)
    _eg_param_2_ =              (.5,     .75,     .5,    0.5,   1.5,   1.8,     3,   .25,    0.50,      .5)
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

        self._name = name
        self.shape = shape
        self.r0 = r0
        # when created by calibrate distortions extra info put here
        self.error = 0.0
        self.premium_target = 0.0
        self.assets = 0.0
        self.mass = 0.0
        self.df = df
        self.col_x = col_x
        self.col_y = col_y
        self.display_name = display_name
        g_prime = None

        # now make g and g_inv
        if self._name == 'ph':
            rho = self.shape
            rhoinv = 1.0 / rho
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
                alpha = np.nan
                self.has_mass = True
                self.mass = 1

                def g(x):
                    return np.where(x==0, 0, 1)

                def g_inv(x):
                    return np.where(x < 1, x * (1 - p), 1)

                def g_prime(x):
                    return np.where(x < 1 - p, alpha, 0)

            else:
                alpha = 1 / (1 - p)
                self.has_mass = False

                def g(x):
                    return np.minimum(alpha * x, 1)

                def g_inv(x):
                    return np.where(x < 1, x * (1 - p), 1)

                def g_prime(x):
                    return np.where(x < 1 - p, alpha, 0)


        elif self._name == 'ly':
            # linear yield
            # r0 = occupancy; rk = consumption specified in list shape parameter
            rk = self.shape
            self.has_mass = (r0 > 0)
            self.mass = r0 / (1 + r0)

            def g(x):
                return np.where(x == 0, 0, (self.r0 + x * (1 + rk)) / (1 + self.r0 + rk * x))

            def g_inv(x):
                return np.maximum(0, (x * (1 + self.r0) - self.r0) / (1 + rk * (1 - x)))

        elif self._name == 'clin':
            # capped linear, needs shape > 1 to make sense...needs shape >= 1-r0 else
            # problems at 1
            sl = self.shape
            self.has_mass = (r0 > 0)
            self.mass = r0

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
            x0, x1 = df
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
                p0, p1 = df
                assert p0 < p1, 'Error: p0 must be less than p1'
                w = shape
                self.has_mass = (p1 == 1)
                self.mass = w if p1 == 1 else 0
                pt = (1 - p1) / (1 - p0) * (1 - w) + w
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
            ps = np.array(shape)
            wts = np.array(df)
            if 1 in ps and wts[-1] > 0:
                self.has_mass = True
                self.mass = wts[-1]
            else:
                self.has_mass = False
                self.mass = 0

            if display_name != '':
                self.display_name = display_name
            else:
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
            self.has_mass = np.all([d.has_mass for d in shape])
            if self.has_mass:
                self.mass = np.min([d.mass for d in shape])
            else:
                self.mass = 0

            if display_name != '':
                self.display_name = display_name
            else:
                self.display_name = f'Minimum of {len(shape):d} distortions'

            loc_shape = shape.copy() ## ? insulates from shape?
            def g(x):
                g_values = np.array([gi.g(x) for gi in loc_shape])
                return np.min(g_values, axis=0)

            def g_prime(x):
                g_values = np.array([gi.g(x) for gi in loc_shape])
                # Find the index of the minimum value
                min_index = np.argmin(g_values, axis=0)
                # Return the derivative (g_prime) of the object that achieved the minimum
                return np.array([loc_shape[i].g_prime(x) for i in min_index])

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

            self.has_mass = np.any([d.has_mass for d in shape])
            if self.has_mass:
                self.mass = np.sum([d.mass * w for (d, w) in zip(shape, df)])
            else:
                self.mass = 0

            if display_name != '':
                self.display_name = display_name
            else:
                self.display_name = f'Mixture of {len(shape):d} distortions'

            loc_shape = shape.copy() ## ? insulates from shape?
            if df is None:
                loc_wts = self.df = np.array([1 / len(loc_shape)] * len(loc_shape))
            else:
                loc_wts = np.array(df.copy())
            def g(x):
                g_values = np.array([gi.g(x) for gi in loc_shape])
                # at this point, have to be a bit careful about the shape of the arrays
                if g_values.ndim > 2:
                    # Flatten the higher dimensions (n1, n2) into a single dimension
                    vals_flat = g_values.reshape(2, -1)  # shape (2, n1 * n2)

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
                    vals_flat = g_values.reshape(2, -1)
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
            self.display_name = f'Convex on {len(df):d} points'
            if not (0 in df[col_x].values and 1 in df[col_x].values):
                # painful...always want 0 and 1 there...but don't know what other columns in df
                # logger.debug('df does not contain s=0/1...adding')
                df = df[[col_x, col_y]].copy().reset_index(drop=True)
                df.loc[len(df)] = (0,0)
                df.loc[len(df)] = (1,1)
                df = df.sort_values(col_x)
            if len(df) > 2:
                hull = ConvexHull(df[[col_x, col_y]])
                knots = list(set(hull.simplices.flatten()))
                g = interp1d(df.iloc[knots, df.columns.get_loc(col_x)],
                             df.iloc[knots, df.columns.get_loc(col_y)], kind='linear',
                             bounds_error=False, fill_value=(0,1))
                g_inv = interp1d(df.iloc[knots, df.columns.get_loc(col_y)],
                             df.iloc[knots, df.columns.get_loc(col_x)], kind='linear',
                             bounds_error=False, fill_value=(0,1))
            else:
                df = df.sort_values(col_x)
                g = interp1d(df[col_x], df[col_y], kind='linear',
                             bounds_error=False, fill_value=(0,1))
                g_inv = interp1d(df[col_y], df[col_x], kind='linear',
                             bounds_error=False, fill_value=(0,1))
        else:
            raise ValueError(
                "Incorrect spec passed to Distortion; implemented g types are ph, wang, tvar, "
                "ly (linear yield), lep (layer equivalent pricing) and clin (clipped linear)")

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
        Utility method to create bitvar.
        """
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
    def minimum(distortion_list, weights=None):
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

    def __str__(self):
        if self.display_name != '':
            s = self.display_name
            return s
        else:
            return self.__repr__()
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

        return s

    def __repr__(self):
        # Get the default __repr__ output
        default_repr = super().__repr__()
        return f'{self.name}: {default_repr}'
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
        if both:
            y2 = self.g_dual(xs)

        if ax is None:
            sz = FIG_H if size=='small' else FIG_W
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
    def test(cls, r0=0.035, df=[0.0, .9]):
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
            ma = f', ms={mass:.3f}'
        if mean:
            ps = np.insert(ps, 0, 0)
            wts = np.insert(wts, 0, mean)
            mn = f', mn={mean:.3f}'
        return Distortion('wtdtvar', ps, df=wts, display_name=f'Rnd {n_knots} knots{mn}{ma}')

    def price(self, ser, a=np.inf, kind='ask', S_calculation='forwards'):
        r"""
        Compute the bid and ask prices for the distribution determined by ``ser`` with
        an asset limit ``a``. Index of ``ser`` need not be equally spaced, so it can
        be applied to :math:`\kappa`. To do this for unit A in portfolio port::

            ser = port.density_df[['exeqa_A', 'p_total']].\
                set_index('exeqa_A').groupby('exeqa_A').\
                sum()['p_total']
            dist.price(ser, port.q(0.99), 'both')

        Always use ``S_calculation='forwards`` method to compute S = 1 - cumsum(probs).
        Computes the price as the integral of gS.

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

        # apply limit
        if a < np.inf:
            # ser = ser.copy()
            # tail = ser[a:].sum()
            ser = ser[:a]
            # ser[a] = tail

        # unlikely to be an issue
        ser = ser.sort_index(ascending=True)

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
        Compute the bid and ask prices for the distribution determined by ``ser`` with
        an asset limits given by values of ``ser``. Index of ``ser`` need not be equally
        spaced, so it can be applied to :math:`\kappa`. To do this for unit A in portfolio
        ``port``::

            ser = port.density_df[['exeqa_A', 'p_total']].\
                set_index('exeqa_A').groupby('exeqa_A').\
                sum()['p_total']
            dist.price(ser, port.q(0.99))

        :param ser: pd.Series of is probabilities, indexed by outcomes. Outcomes must
          be spaced evenly. ``ser`` is usually a probability column from ``density_df``.
        """

        if not isinstance(ser, pd.Series):
            raise ValueError(f'ser must be a pandas Series, not {type(ser)}')

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

    @staticmethod
    def calibrate(self, name, premium_target, density_df, assets=np.inf, df=None):
        """
        TODO: TEST THIS!! 
        Find transform to hit a premium target.
        Based off Portfolio.calibrate_distortion, without many of the options.
        Assumes cumsum S_calc method. No funky adjustments. Assumes S is passed in as
        argument.

        :param name: type of distortion
        :param premium_target: target premium
        :param density_df: from Portfolio, Aggregate, or similar. Needs xs index and S column
        :param assets: optional asset level, default unlimited (ccoc not available)
        :param df: [p0, p1] for BiTVaR distortion
        :return: newly created Distortion object achieving desired pricing.
        """

        Splus = (1 - density_df.loc[0:assets, 'p_total'].cumsum()).values
        bs = density_df.index[1] - density_df.index[0]

        last_non_zero = np.argwhere(Splus)
        ess_sup = 0
        if len(last_non_zero) == 0:
            # no nonzero element
            last_non_zero = len(Splus) + 1
        else:
            last_non_zero = last_non_zero.max()
        # remember length = max index + 1 because zero based
        if last_non_zero + 1 < len(Splus):
            # now you have issues...
            # truncate at first zero; numpy indexing because values
            S = Splus[:last_non_zero + 1]
            ess_sup = density_df.index[last_non_zero + 1]
            logger.info(
                'Portfolio.calibrate_distortion | Mass issues in calibrate_distortion...'
                f'{name} at {last_non_zero}, loss = {ess_sup}')
            print('Triggering WEIRD CODE in calibrate.')
        else:
            S = (1 - density_df.loc[0:assets - bs, 'p_total'].cumsum()).values

        # now all S values should be greater than zero  and it is decreasing
        assert np.all(S > 0) and np.all(S[:-1] >= S[1:])

        if name == 'ph':
            lS = np.log(S)
            shape = 0.95  # starting param

            def f(rho):
                trho = S ** rho
                ex = np.sum(trho) * bs
                ex_prime = np.sum(trho * lS) * bs
                return ex - premium_target, ex_prime
        elif name == 'wang':
            n = ss.norm()
            shape = 0.95  # starting param

            def f(lam):
                temp = n.ppf(S) + lam
                tlam = n.cdf(temp)
                ex = np.sum(tlam) * bs
                ex_prime = np.sum(n.pdf(temp)) * bs
                return ex - premium_target, ex_prime
        elif name == 'ly':
            # linear yield model; min rol is ro/(1+ro)
            shape = 1.25  # starting param
            mass = ess_sup * r0 / (1 + r0)

            def f(rk):
                num = r0 + S * (1 + rk)
                den = 1 + r0 + rk * S
                tlam = num / den
                ex = np.sum(tlam) * bs + mass
                ex_prime = np.sum(S * (den ** -1 - num / (den ** 2))) * bs
                return ex - premium_target, ex_prime
        elif name == 'clin':
            # capped linear, input rf as min rol
            shape = 1
            mass = ess_sup * r0

            def f(r):
                r0_rS = r0 + r * S
                ex = np.sum(np.minimum(1, r0_rS)) * bs + mass
                ex_prime = np.sum(np.where(r0_rS < 1, S, 0)) * bs
                return ex - premium_target, ex_prime
        elif name in ['roe', 'ccoc']:
            # constant roe
            # TODO Note if you input the roe this is easy!
            shape = 0.25
            def f(r):
                v = 1 / (1 + r)
                d = 1 - v
                mass = ess_sup * d
                r0_rS = d + v * S
                ex = np.sum(np.minimum(1, r0_rS)) * bs + mass
                ex_prime = np.sum(np.where(r0_rS < 1, S, 0)) * bs
                return ex - premium_target, ex_prime
        elif name == 'lep':
            # layer equivalent pricing
            # params are d=r0/(1+r0) and delta* = r/(1+r)
            d = r0 / (1 + r0)
            shape = 0.25  # starting param
            # these hard to compute variables do not change with the parameters
            rSF = np.sqrt(S * (1 - S))
            mass = ess_sup * d

            def f(r):
                spread = r / (1 + r) - d
                temp = d + (1 - d) * S + spread * rSF
                ex = np.sum(np.minimum(1, temp)) * bs + mass
                ex_prime = (1 + r) ** -2 * np.sum(np.where(temp < 1, rSF, 0)) * bs
                return ex - premium_target, ex_prime
        elif name == 'tt':
            assert name != 'tt', 'Not implemented for tt distortion because it is not a distortion.'
            # wang-t-t ... issue with df, will set equal to 5.5 per Shaun's paper
            # finding that is a reasonable level; user can input alternative
            # param is shape like normal
            t = ss.t(df)
            shape = 0.95  # starting param

            def f(lam):
                temp = t.ppf(S) + lam
                tlam = t.cdf(temp)
                ex = np.sum(tlam) * bs
                ex_prime = np.sum(t.pdf(temp)) * bs
                return ex - premium_target, ex_prime
        elif name == 'cll':
            # capped loglinear
            shape = 0.95  # starting parameter
            lS = np.log(S)
            lS[0] = 0
            ea = np.exp(r0)

            def f(b):
                uncapped = ea * S ** b
                ex = np.sum(np.minimum(1, uncapped)) * bs
                ex_prime = np.sum(np.where(uncapped < 1, uncapped * lS, 0)) * bs
                return ex - premium_target, ex_prime
        elif name == 'dual':
            # dual moment
            # be careful about partial at s=1
            shape = 2.0  # starting parameter
            lS = -np.log(1 - S)  # prob a bunch of zeros...
            lS[S == 1] = 0

            def f(rho):
                temp = (1 - S) ** rho
                trho = 1 - temp
                ex = np.sum(trho) * bs
                ex_prime = np.sum(temp * lS) * bs
                return ex - premium_target, ex_prime
        elif name == 'tvar':
            # tvar
            shape = 0.9   # starting parameter
            def f(rho):
                temp = np.where(S <= 1-rho, S / (1 - rho), 1)
                temp2 = np.where(S <= 1-rho, S / (1 - rho)**2, 1)
                ex = np.sum(temp) * bs
                ex_prime = np.sum(temp2) * bs
                return ex - premium_target, ex_prime

        elif name == 'wtdtvar':
            # weighted tvar with fixed p parameters
            shape = 0.5  # starting parameter
            p0, p1 = df
            s = np.array([0., 1-p1, 1-p0, 1.])

            def f(w):
                pt = (1 - p1) / (1 - p0) * (1 - w) + w
                gs = np.array([0.,  pt,   1., 1.])
                g = interp1d(s, gs, kind='linear')
                trho = g(S)
                ex = np.sum(trho) * bs
                ex_prime = (np.sum(np.minimum(S / (1 - p1), 1) - np.minimum(S / (1 - p0), 1))) * bs
                return ex - premium_target, ex_prime
        else:
            raise ValueError(f'calibrate_distortion not implemented for {name}')

        # numerical solve except for tvar, and roe when premium is known
        if name in ('roe', 'ccoc'):
            assert assets < np.inf, f'Must input finite assets for ccoc.'
            el = np.sum(density_df.index * density_df.p_total)
            r = (premium_target - el) / (assets - premium_target)
            shape = r
            r0 = 0
            fx = 0
        else:
            i = 0
            fx, fxp = f(shape)
            max_iter = 200 if name == 'tvar' else 50
            while abs(fx) > 1e-5 and i < max_iter:
                shape = shape - fx / fxp
                fx, fxp = f(shape)
                i += 1
            if abs(fx) > 1e-5:
                logger.warning(
                    f'Distortion.calibrate | Questionable convergence for {name} distortion, target '
                    f'{premium_target} error {fx} after {i} iterations')

        # build answer
        dist = Distortion(name=name, shape=shape, r0=r0, df=df)
        dist.error = fx
        dist.assets = assets
        dist.premium_target = premium_target
        return dist


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