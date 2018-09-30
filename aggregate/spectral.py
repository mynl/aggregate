"""
Distortion functions to implement spectral risk measures
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from io import StringIO
import pandas as pd
from . utils import axiter_factory, suptitle_and_tight


class Distortion(object):
    """
    handles everything to do with distortion functions


    """
    # make these (mostly) immutable...avoid changing by mistake
    _available_distortions_ = ('ph', 'wang', 'lep', 'ly', 'clin', 'tvar', 'convex')
    _long_names_ = ("Proportional Hazard", "Wang-normal", "Layer Equivalent Pricing", "Linear Yield", "Capped Linear",
                    "Tail VaR", "Convex Envelope")
    _eg_param_1_ = (.9, 1, 0.25, 0.9, 1.1, 0.75)
    _eg_param_2_ = (.5, 2, 0.35, 1.5, 1.8, 0.95)
    _distortion_names_ = dict(zip(_available_distortions_, _long_names_))

    @classmethod
    def available_distortions(cls, pricing=True):
        """
        list of the available distortions

        :param pricing: only return list suitable for pricing
        :return:
        """

        if pricing:
            return cls._available_distortions_[:-2]
        else:
            return cls._available_distortions_

    def __init__(self, name, shape, r0=0.0, df=None, col_x='', col_y=''):
        """
        create new distortion

        :param name: name of an available distortion, call ``Distortion.available_distortions()`` for a list
        :param shape: float or [float, float]
        :param shape: shape parameter
        :param r0: risk free or rental rate of interest
        :param df:  for convex envelope, dataframe with col_x and col_y used to parameterize
        :param col_x:
        :param col_y:
        """
        self.name = name
        self.shape = shape
        self.r0 = r0
        # when created by calibrate distortions extra info put here
        self.error = 0.0
        self.premium_target = 0.0
        self.assets = 0.0
        self.mass = 0.0

        # now make g and g_inv
        if self.name == 'ph':
            rho = self.shape
            rhoinv = 1.0 / rho
            self.has_mass = False

            # @numba.vectorize(["float64(float64)"], nopython=True, target='parallel')
            def g(x):
                return x ** rho

            def g_inv(x):
                return x ** rhoinv

        elif self.name == 'wang':
            lam = self.shape
            n = ss.norm()
            self.has_mass = False

            def g(x):
                return n.cdf(n.ppf(x) + lam)

            def g_inv(x):
                return n.cdf(n.ppf(x) - lam)

        elif self.name == 'tvar':
            p = self.shape
            alpha = 1 / (1 - p)
            self.has_mass = False

            def g(x):
                return np.minimum(alpha * x, 1)

            def g_inv(x):
                return np.where(x < 1, x * (1 - p), 1)

        elif self.name == 'ly':
            # linear yield
            # r0 = occupancy; rk = consumption specified in list shape parameter
            rk = self.shape
            self.has_mass = (r0 > 0)
            self.mass = r0 / (1 + r0)

            def g(x):
                return np.where(x == 0, 0, (self.r0 + x * (1 + rk)) / (1 + self.r0 + rk * x))

            def g_inv(x):
                return np.maximum(0, (x * (1 + self.r0) - self.r0) / (1 + rk * (1 - x)))

        elif self.name == 'clin':
            # capped linear
            sl = self.shape
            self.has_mass = (r0 > 0)
            self.mass = r0

            def g(x):
                return np.where(x == 0, 0, np.minimum(1, self.r0 + sl * x))

            def g_inv(x):
                return np.where(x <= self.r0, 0, (x - self.r0) / sl)

        elif self.name == 'lep':
            # layer equivalent pricing
            # self.r0 = risk free/financing and r = risk charge (the solved parameter)
            r = self.shape
            delta = r / (1 + r)
            d = self.r0 / (1 + self.r0)
            spread = delta - d
            self.has_mass = (d > 0)
            self.mass = d

            def g(x):
                return np.minimum(1, d + (1 - d) * x + spread * np.sqrt(x * (1 - x)))

            # kludge for now
            # TODO sort out g_inv...it is incorrect...
            # g, g_inv = agg.distortion_factory({'name': 'lep', 'shape': [0.05, 0.35]})
            # ps = np.linspace(0, 1, 1001)
            # df = pd.DataFrame({'log': ps, 'gg_inv': g(g_inv(ps)), 'g_invg': g_inv(g(ps)),
            # 'g': g(ps), 'g_inv': g_inv(ps)})
            # df.query(' abs(gg_inv - g_invg) > 1e-5')
            sigma = (delta - d) ** 2
            a = (1 - d) ** 2 + sigma

            def g_inv(x):
                mb = (2 * (x - d) * (1 - d) + sigma)  # mb = -b
                c = (x - d) ** 2
                rad = np.sqrt(mb * mb - 4 * a * c)
                # l = (mb + rad)/(2 * a)
                u = (mb - rad) / (2 * a)
                return np.where(x < d, 0, np.maximum(0, u))

        elif self.name == 'convex':
            self.has_mass = False
            hull = ConvexHull(df[[col_x, col_y]])
            knots = list(set(hull.simplices.flatten()))
            g = interp1d(df.iloc[knots, df.columns.get_loc(col_x)],
                         df.iloc[knots, df.columns.get_loc(col_y)], kind='linear')
            g_inv = interp1d(df.iloc[knots, df.columns.get_loc(col_y)],
                         df.iloc[knots, df.columns.get_loc(col_x)], kind='linear')
        else:
            raise ValueError(
                "Incorrect spec passed to distortion_factory; implemented g types are ph, wang, tvar, "
                "ly (linear yield), lep (layer equivalent pricing) and clin (clipped linear)")

        self.g = g
        self.g_inv = g_inv

    def __str__(self):
        """
        printable version of distortion

        :return:
        """
        if isinstance(self.shape, str):
            s = f'{self._distortion_names_[self.name]}\n{self.shape}'
        else:
            s = f'{self._distortion_names_[self.name]}\n{self.shape:.3f}'
        if self.has_mass:
            s += f', {self.r0:.3f}'
        # else:
        #     s += ')'
        return s

    def __repr__(self):
        s = f'{self.name} ({self.shape}'
        if self.has_mass:
            s += f', {self.r0})'
        else:
            s += ')'
        return s

    # noinspection PyUnusedLocal
    def plot(self, xs=None, n=101, both=True, ax=None, **kwargs):
        """
        quick plot of the distortion

        :param ax:
        :param xs:
        :param n:  length of vector is no xs
        :param both: ignored for now. just do both
        :param kwargs:  passed to plot
        :return:
        """

        if xs is None:
            xs = np.linspace(0, 1, n)

        y1 = self.g(xs)
        y2 = self.g_inv(xs)

        if ax is None:
            ax = plt.gca()

        ax.plot(xs, y1, **kwargs)
        ax.plot(xs, y2, **kwargs)
        ax.plot(xs, xs, lw=0.5, color='black', alpha=0.5)
        ax.grid(which='major', axis='both', linestyle='-', linewidth='0.1', color='blue', alpha=0.5)
        ax.set_title(self.__str__())

    @classmethod
    def test(cls, r0=0.05):
        """
        tester: make some nice plots

        :return:
        """

        axiter = axiter_factory(None, 12, figsize=(10, 6))

        xs = np.linspace(0, 1, 1001)

        for name, shape in zip(cls._available_distortions_, cls._eg_param_1_):
            dist = Distortion(name, shape, r0)
            dist.plot(xs, ax=next(axiter))

        for name, shape in zip(cls._available_distortions_, cls._eg_param_2_):
            dist = Distortion(name, shape, r0)
            dist.plot(xs, ax=next(axiter))

        dist = Distortion.convex_example('yield')
        dist.plot(xs, ax=next(axiter))

        dist = Distortion.convex_example('cat')
        dist.plot(xs, ax=next(axiter))

        axiter.tidy()
        suptitle_and_tight('Example Distortion Functions')

    @staticmethod
    def distortions_from_params(params, index, r0=0.025, plot=True):
        """
        make set of dist funs and inverses from params, output of port.calibrate_distortions
        params must just have one row for each method and be in the output format of cal_dist

        :param plot:
        :param index:
        :param r0: min rol parameters
        :param params: dataframe such that params[index, :] has a [lep, param] etc.
        :return:
        """
        temp = params.loc[index, :]
        dists = {}
        for dn in Distortion.available_distortions():
            param = float(temp.loc[dn, 'param'])
            dists[dn] = Distortion(name=dn, shape=param, r0=r0)

        if plot:
            axiter = axiter_factory(None, 5)
            # f, axs = plt.subplots(2, 3, figsize=(8, 6))
            # it = iter(axs.flatten())
            for dn in Distortion.available_distortions():
                dists[dn].plot(ax=next(axiter))
            axiter.tidy()
            plt.tight_layout()

        return dists  # [g_lep, g_ph, g_wang, g_ly, g_clin]

    @staticmethod
    def convex_example(source='yield'):
        """
        example convex distortion using data from https://www.bis.org/publ/qtrpdf/r_qt0312e.pdf

        :param source: yield gives yield curve example, cat gives cat bond / cat reinsurance pricing based example
        :return:
        """

        if source == 'yield':
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