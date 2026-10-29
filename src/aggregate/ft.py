"""Direct inversion of characteristic functions: a forwarding shim.

This module is superseded by :mod:`aggregate.characteristic_function`, whose
:class:`~aggregate.characteristic_function.AggregateCHF` returns bucket masses
rather than density samples, sizes its own window from the law's tails, tilts
a heavy-tailed law along a shifted contour so that it does not wrap, and is a
first-class object with the DecL statement ``chfagg``
(``dev/done/plan-characteristic-function-engine.md``). Where each name went:

* ``FourierTools(chf, fz)`` with a known law: ``build('chfagg ...')`` or
  ``AggregateCHF.from_family``;
* ``FourierTools(chf, 'continuous')``, an arbitrary ``chf``:
  ``AggregateCHF.from_chf``;
* ``invert`` (density samples): ``update`` (bucket masses; ``pdf`` is the mass
  over ``bs``);
* ``invert_simpson``: ``pedagogy.chf_simpson_inversion``;
* ``compute_exact``: ``pedagogy.chf_exact_comparison``;
* ``plot``, ``plot_wraps``, ``plot_simpson``, ``plot_fourier3da``:
  ``pedagogy.chf_inversion_figure``, ``chf_wrapping_figure``,
  ``chf_simpson_figure``, ``chf_fourier3d_figure``;
* ``make_levy_chf``: ``Stable(alpha, beta).chf``;
* ``poisson_example``, ``fft_wrapping_illustration``,
  ``recentering_convolution_example``, ``recentering_convolution``, ``stats``:
  moved unchanged to :mod:`aggregate.pedagogy`.

``FourierTools`` keeps its body unchanged, so old notebooks reproduce their
numbers, and its constructor raises a ``DeprecationWarning``; the moved names
are forwarded through a module ``__getattr__`` that warns once per name.
Nothing inside the library imports this module. Removal is
``[FourierTools-Shim-Removal]`` in ``dev/TODO.md``: when the outside
documents that use it have moved, or at the next major release.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.fft import irfft, rfft, ifft as ift
from numpy import roll

from .constants import FIG_H, FIG_W
from .distributions import Aggregate
from .underwriter import build
from .utilities import qd, remove_fuzz as remove_fuzz_util

logger = logging.getLogger(__name__)


class FourierTools:
    """Manual inversion of a ch. f. using FFTs."""

    def __init__(self, chf, fz, scale_mode=True):
        """
        Class version of manual inversion of characteristic function.

        Numerical inversion is split across four methods:

        1. ``invert`` — the FFT inversion itself.
        2. ``compute_exact`` — exact density via ``fz`` (when available).
        3. ``plot`` — graph to compare densities.
        4. ``plot_wraps`` — graph to compute effect of wrapping (aliasing).

        For discrete rvs, x_max is always n - 1 and the bucketr size 1.
        For continuous rvs, it is either input or estimated as a quantile.

        The arguments completely define the distribution of interest. Other class
        functions vary the numerical variables, defining the window and number of
        points used in the FFT routine.

        See BLOG POST.

        :param chf: the characteristic function of the distribution, takes args t;
          routine handles conversion to Fourier Transform and adds loc and scale effects.
          Must use same shape parameters as fz.
        :param fz: the scipy.stats frozen distribution object. Used to compute the exact answer.
          If fz is 'discrete' or 'continuous' or 'mixed' it is a generic distribution
          with no closed form cdf/pdf, e.g. Tweedie. Then you can't compute exact,
          obviously.
        :param scale_mode: if True, the scale parameter from fz is used in the Fourier
          Transform, otherwise it is unadjusted.
        """
        warnings.warn(
            'FourierTools is superseded by AggregateCHF.from_chf and '
            'AggregateCHF.from_family (aggregate.characteristic_function); '
            'aggregate.ft is a forwarding shim ([FourierTools-Shim-Removal])',
            DeprecationWarning, stacklevel=2)
        self.chf = chf
        self.fz = fz
        self.scale_mode = scale_mode
        if isinstance(fz, str):
            # extremely limited functionality
            self.params = (0,)  # allows passing as param 1 to the chf
            self.loc = 0.
            self.scale = 1.
            self.discrete = True if fz == 'discrete' else False
            self.distribution_name = 'User defined'
        elif isinstance(fz, Aggregate):
            self.params = (0,)
            self.loc = 0.
            self.scale = 1.
            self.discrete = fz.bs == 1
            self.distribution_name = f'Aggregate({fz.name})'
        elif fz is None:
            # flying blind...do the best we can
            self.params = (0,)
            self.loc = 0.
            self.scale = 1.
            self.discrete = False
            self.distribution_name = 'Unknown'
        else:
            # extract shape params from fz; no longer used
            self.params = fz.args
            # location and scale with default 0, 1
            kwds = fz.kwds
            self.loc = kwds.get('loc', 0.)
            self.scale = kwds.get('scale', 1)
            # is this a discrete or continuous variable?
            self.discrete = True if str(type(fz)).find('discrete') > 0 else False
            self.distribution_name = fz.dist.name
        # slightly ugly state, but this encodes n=2**log2, xmin, xmax, etc.
        self._df = self._df_exact = None
        self._ts = None      # phases (angles) at which ft evaluated , for plotting
        self._fourier = None  # store the sample of ft at self._ts for plotting
        self.last_fig = None
        # from the last run, note: x_range = P
        self.bs = self.x_range = self.x_min = self.x_max = self.log2 = 0
        self.exact_calc = ""

    def __repr__(self):
        """Repr of the object."""
        return f'FourierTools({self.distribution_name}{self.params}, loc={self.loc}, scale={self.scale})'

    def describe(self):
        """More information."""
        return f'{repr(self)}\nn={2**self.log2}, x_min={self.x_min}, x_max={self.x_max:.3g}, bs={self.bs:.3g}'

    @property
    def df(self):
        """Return current state dataframe output."""
        if self._df is not None:
            return self._df
        else:
            raise ValueError('Must run invert first!')

    @property
    def df_exact(self):
        """Return current state dataframe output."""
        if self._df is not None:
            return self._df_exact
        else:
            raise ValueError('Must run compute_exact first!')

    def fourier_transform(self, t):
        """
        Create ft function  by converting ch f to Fourier transform and including scale and loc.

        Recall:
        ft(t)= int f(x)exp(-2πi t x)dx
        chf(t) = int f(x)exp(i t x)dx.
        """
        TWOPI = 6.283185307179586
        t1 = -t * TWOPI
        if self.scale_mode:
            ans = self.chf(t1 * self.scale)
            if self.loc != 0:
                # for some reason ans *= np.exp(-t1 * loc) does not work
                ans = ans * np.exp(t1 * self.loc * 1j)
        else:
            ans = self.chf(t1)
        return ans

    def invert(self, log2, x_min=0, bs=0, x_max=None, s=1e-17):
        """
        Invert a characteristic function using irfft.

        Call with just log2 for positive support, to determine x_max and
        bs from quantiles. Call with bs fixed to a reasonable value to
        deduce x_max = x_min + n bs. Call with x_max to fix the range
        and deduce bs.

        :param x_min: minimum value of range.
        :param bs: bucket size to use, determines x_max
        :param x_max: minimum value of range.
        :param s: survival probability to determine tails
        """
        # number of buckets
        self.log2 = log2
        n = 1 << log2
        if x_min is None:
            x_min = self.fz.ppf(s)
        if bs == 0:
            if self.discrete:
                x_max = x_min + n
            else:
                x_max = self.fz.isf(s) if x_max is None else x_max
        else:
            x_max = x_min + bs * n
        if x_min == 0 and x_max == 0:
            raise ValueError('Must provide x_min < 0 or x_max > 0. Current range is 0 to 0!')

        # spatial range is [x_min, x_max]
        # translate to [0, x_max - x_min]
        x_range = x_max - x_min
        # sampling interval (wavelength) = bs = x_range / n
        # sampling domain, for exact and to "label" the Fourier Transform output
        # sampling interval is bs (small bs means high sampling rate)
        # the highest sampling freq for inverting the FT is 1 / bs = n / x_range
        bs = x_range / n
        if self.discrete and bs != 1:
            logger.warning(f'{bs=}, not the expected 1 for a discrete rv')

        # f_max is 1 / bs
        f_max = 1 / bs
        # f(x) = int_R fhat(t) exp(2πi tx)dt ≈ int_-f_max_f^max_f ...
        # sample the FT; using real fft, only need half the range
        # self._ts = np.arange(n // 2 + 1) * f_max / n
        self._ts = np.linspace(0, 1/2, n // 2 + 1) * f_max
        self._fourier = self.fourier_transform(self._ts)
        probs = irfft(self._fourier)
        if x_min != 0:
            probs = np.roll(probs, -int(x_min / bs))

        # for df index
        self._df = pd.DataFrame({
            'x': np.linspace(x_min, x_max, n, endpoint=False),
            'p': probs}).set_index('x')

        # store for future use
        self.bs, self.x_range, self.x_min, self.x_max = bs, x_range, x_min, x_max

    def invert_simpson(self, *, log2=0, bs=0, x_min=None):
        """
        Add Simpson's method approximation.

        Adds ``simpson`` column to
        ``self.df``. Uses method of Wang and Zhang, "Simpson's rule based FFT
        method to compute densities of stable distribution" (2008). Can no
        longer use the real fft and ifft methods because the input vector is
        not conjugate symmetric about its midpoint.

        Run after self.invert to set parameters or input directly. Defaults
        are None because 0 is a legitimate value for x_min. Convenient to
        state x_min and bs directly. Best to input all parameters if adjusting
        to avoid mismatches. Note x_max gets trumped by n, bs, and x_min.
        """
        # parameters to use for the calculation
        log2 = log2 if log2 > 0 else self.log2
        n = 1 << log2
        x_min = x_min if x_min is not None else self.x_min
        bs = bs if bs > 0 else self.bs
        f_max = 1 / bs
        ks = np.arange(0, n)
        P = n * bs   # period

        t_left = np.linspace(-f_max / 2, f_max / 2, n, endpoint=False)
        t_left = np.roll(t_left, n >> 1)

        if x_min != 0:
            # pre-FFT adjustment
            x_min_adj = np.exp(2 * np.pi * 1j * x_min * t_left)
            # post-FFT adjustment
            post_2 = np.exp(2 * np.pi * 1j * x_min / (2 * P))
            post_3 = np.exp(2 * np.pi * 1j * x_min / P)
        else:
            # no adjustments needed
            x_min_adj = post_2 = post_3 = 1.

        phi_left = self.fourier_transform(t_left)
        # three terms left, middle, right of simpson's rule
        term1 = ift(x_min_adj * phi_left)
        term2 = post_2 * np.exp(np.pi * 1j * ks / n) * ift(
            x_min_adj * self.fourier_transform(t_left + 0.5 / n * f_max))
        term3 = post_3 * np.exp(2 * np.pi * 1j * ks / n) * ift(
            x_min_adj * self.fourier_transform(t_left + 1 / n * f_max))
        # weighted sum
        simpson = (term1 + 4. * term2 + term3) / 6.
        # check imaginary part small
        max_abs = np.abs(np.imag(simpson)).max()
        if max_abs > 1e-10:
            logger.warning(f'Answer has suspiciously large imaginary component {max_abs}')
        # create / update answer dataframe
        if self._df is None or self.bs != bs or self.x_min != x_min:
            # changed scale or location, recreate dataframe from scratch
            logger.info('recreating df')
            self._df = pd.DataFrame({
                'x': np.linspace(x_min, x_min + n * bs, n, endpoint=False),
                'simpson': np.real(simpson)}).set_index('x')
        else:
            # append to existing dataframe
            self.df['simpson'] = np.real(simpson)

    def compute_exact(self, calc='survival', max_points=257):
        """
        Compute exact density using frozen scipy.stats object.

        :param calc: 'density' re-scales pdf, 'survival' uses backward
          differences of sf.
        :param max_points: maximum number of points to use; if more just
          interpolate this many points.
        :param decimate: decimate (take ``::decimate``) input xs to reduce
          number of calls to fz (which may be slow)
        :param min_points: opposite of decimate, ensure exact computed
          with min_points. Useful for example when log2 is "too small"
          to ensure exact distribution is rendered correctly.
        """
        assert calc in ('survival', 'density'), 'calc must be "survival" or "density"'
        self.exact_calc = calc
        assert self._df is not None, 'Must recompute first. Run invert().'
        xs = np.array(self._df.index)
        if len(xs) > max_points and not self.discrete:
            # mostly this is an issue for non-discrete distributions
            xs = np.linspace(xs[0], xs[-1] + self.bs, max_points)
        # if decimate > 1:
        #     xs = xs[::decimate]
        # # if too few points beef up, non-discrete distributions only
        # if len(xs) < min_points and not self.discrete:
        #     xs = np.linspace(xs[0], xs[-1] + self.bs, min_points)
        self.bs_exact = bs = xs[1] - xs[0]
        # discrete dists have pmf not pdf
        if getattr(self.fz, 'pdf', None) is None:
            pdf = self.fz.pmf
        else:
            pdf = self.fz.pdf

        if calc == 'density':
            logger.warning('Best to use survival calc rather than density method.')
            exact = pdf(xs)
            self._df_exact = pd.DataFrame({'x': xs, 'p': exact}).set_index('x')
        elif calc == 'survival':
            xs1 = np.hstack((xs - bs / 2, xs[-1] + bs / 2))
            exact = -np.diff(self.fz.sf(xs1)) /  bs
            self._df_exact = pd.DataFrame({'x': xs, 'p': exact}).set_index('x')
        # self.decimate = decimate
        return self._df_exact

    def plot(self, suptitle='', xlim=None, verbose=True):
        """
        Compare density, log density, and plot amplitude and argument of Fourier transform.

        :param suptitle: super title for the plot.
        """
        from .plots import plot_fourier
        return plot_fourier(self, suptitle=suptitle, xlim=xlim, verbose=verbose)

    def plot_wraps(self, wraps=None, calc='survival', add_tail=False):
        """
        Illustrate wrapping. Only run when fz.pdf is easy to calc, otherwise too slow.

        :param wraps: optional list of wrap values. Eg [-1,1] plots one greater and one
            less than [0, P)
        :param calc: how to estimate the density outside the base range, same ``compute_exact``.
        :param add_tail: plot the shifted exact densities in plots 2 and 4.
        """
        from .plots import plot_fourier_wraps
        return plot_fourier_wraps(self, wraps=wraps, calc=calc, add_tail=add_tail)

    def plot_simpson(self, ylim=1e-16):
        """Plot Simpson's approximation."""
        from .plots import plot_fourier_simpson
        return plot_fourier_simpson(self, ylim=ylim)

    def _plot_fourier1d(self, ax, min_abs=1e-20):
        """Create simple plot of Fourier transform on one axis."""
        from .plots import plot_fourier1d
        return plot_fourier1d(self, ax, min_abs=min_abs)

    def plot_fourier3d(self, scale=True):
        """Three dimensional line plot of the Fourier transform using mayavi."""
        # dont want to make mayavi a required package
        logger.warning('REMEMBER: this routine pops a separate window!')
        try:
            from mayavi import mlab
        except ModuleNotFoundError:
            raise ModuleNotFoundError('mayavi required, pip install mayavi')

        # Generate data
        t = self._ts
        f_t = self._fourier
        c = np.abs(f_t)  # Use |f(t)| for coloring

        if scale:
            f_t = f_t / c
        f_t[np.isinf(f_t)] = np.nan

        x = np.real(f_t)
        y = np.imag(f_t)
        # scale to 0, 1
        z = t / np.max(t)

        # Create the 3D line plot
        mlab.figure(size=(1000, 800))  # Set figure size
        mlab.plot3d(x, y, z, c, tube_radius=0.005, colormap='viridis')

        # Add axes and labels
        mlab.xlabel('Re(f)')
        mlab.ylabel('Im(f)')
        mlab.zlabel('scaled t')
        mlab.title('Scaled FT' if scale else "FT")
        # Show the plot
        mlab.show()

    def plot_fourier3da(self):
        """Three dimensional line plot of the Fourier transform using plotly."""
        # dont want to make plotly a required package
        try:
            import plotly.graph_objects as go
        except ModuleNotFoundError:
            raise ModuleNotFoundError('plotly required, pip install plotly')
        t = self._ts
        f_t = self._fourier
        x = np.real(f_t)
        y = np.imag(f_t)
        c = np.abs(f_t)

        # normalized plot data
        f_t_rhs = f_t / np.abs(f_t)
        x_rhs = np.real(f_t_rhs)
        y_rhs = np.imag(f_t_rhs)

        # Create the subplot figure
        fig = go.Figure()

        # LHS plot
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=t,
                mode="lines",
                line=dict(
                    color=.5,
                    width=4,
                ),
                name="blue: f(t)",
            )
        )

        # RHS plot
        fig.add_trace(
            go.Scatter3d(
                x=x_rhs,
                y=y_rhs,
                z=t,
                mode="lines",
                line=dict(
                    color=c,
                    colorscale="Plasma",
                    width=4,
                    colorbar=dict(
                        title="|f(t)|",
                        len=0.5,
                        lenmode="fraction",
                    ),
                ),
                name="colored: f(t) / |f(t)|",
            )
        )

        # Update layout for side-by-side 3D plots
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Re(f)", range=[-1, 1]),
                yaxis=dict(title="Im(f)", range=[-1, 1]),
                zaxis=dict(title="t", range=[0, .55]),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1)
            ),
            width=900,  # Wider figure for two plots
            height=800,
            title='Fourier transform and normalized transform.'
        )
        return fig


#: The names forwarded to ``aggregate.pedagogy`` (PEP 562), each warning once.
_MOVED = ('poisson_example', 'fft_wrapping_illustration',
          'recentering_convolution_example', 'recentering_convolution', 'stats')
_warned = set()


def __getattr__(name):
    if name in _MOVED:
        if name not in _warned:
            _warned.add(name)
            warnings.warn(
                f'aggregate.ft.{name} has moved to aggregate.pedagogy.{name}; '
                'aggregate.ft is a forwarding shim ([FourierTools-Shim-Removal])',
                DeprecationWarning, stacklevel=2)
        from . import pedagogy
        return getattr(pedagogy, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def make_levy_chf(alpha, beta):
    """The S1 stable characteristic function; superseded by ``Stable(alpha, beta).chf``.

    Returns the ``chf`` method of :class:`aggregate.stable.Stable` at unit
    scale and zero location, which is the function this used to build.
    """
    warnings.warn('make_levy_chf is superseded by Stable(alpha, beta).chf',
                  DeprecationWarning, stacklevel=2)
    from .stable import Stable
    return Stable(alpha, beta).chf
