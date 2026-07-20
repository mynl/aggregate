"""Frequency distributions: the Frequency base class and its kind subclasses.

Extracted from ``distributions.py`` (Phase 1, kind split). Imported through the
``distributions`` facade so every existing import path keeps working.
"""

from functools import cached_property, wraps
import logging
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.special import kv, gammaln, hyp1f1
from scipy.optimize import broyden2, newton_krylov, brentq
from scipy.optimize import NoConvergence  # noqa
from . import tail as _tail

from ._aggregate_compute import evaluate_pgf_polynomial
from ._severity import validate_discrete_distribution

logger = logging.getLogger(__name__)

__all__ = [
    'Frequency',
    'FrequencyPoisson',
    'FrequencyFixed',
    'FrequencyBernoulli',
    'FrequencyBinomial',
    'FrequencyNegbin',
    'FrequencyGeometric',
    'FrequencyLogarithmic',
    'FrequencyNeymanA',
    'FrequencyPascal',
    'FrequencyEmpirical',
    'FrequencyRenewal',
    'FrequencyGammaMixed',
    'FrequencyDelaporteMixed',
    'FrequencyIGMixed',
    'FrequencySIGMixed',
    'FrequencyBetaMixed',
    'FrequencySichel',
    'FrequencySichelGamma',
    'FrequencySichelIG',
]

def _logarithmic_theta(mean):
    """
    Solve for theta parameter given mean, see JKK p. 288
    """
    f = lambda x: x / (-np.log(1 - x) * (1 - x)) - mean
    theta = brentq(f, 1e-10, 1-1e-10)
    if not np.allclose(mean, theta / (-np.log(1 - theta) * (1 - theta))):
        print('num method failed')
    else:
        return theta


def _normalize_freq_name(freq_name):
    """
    Map user-supplied frequency names to registry keys.

    Handles the synonym groups ``'neyman' | 'neymana' | 'neymanA'`` and the
    ``'sichel.gamma'`` / ``'sichel.ig'`` dotted forms. Unknown names pass
    through unchanged.
    """
    if freq_name in ('neyman', 'neymanA'):
        return 'neymana'
    return freq_name


class Frequency(object):
    """
    Manages Frequency distributions: creates moment function and MGF.

    - freq_moms(n): returns EN, EN^2 and EN^3 when EN=n
    - freq_pgf(n, z): returns the moment generating function applied to z when EN=n

    Frequency distributions are either non-mixture types or mixture types.

    **Non-Mixture** Frequency Types

    - ``fixed``: no parameters
    - ``bernoulli``: exp_en interpreted as a probability, must be < 1
    - ``binomial``: Binomial(n/p, p) where p = freq_a, and n = exp_en
    - ``poisson``: Poisson(n)
    - ``geometric``: geometric(1/(n + 1)), supported on 0, 1, 2, ...
    - ``logarithmci``: logarithmic(theta), supported on 1, 2, ...; theta solved numerically
    - ``negymana``: Po(n/freq_a) stopped sum of Po(freq_a) freq_a = "eggs per cluster"
    - ``negbin``: freq_a is the variance multiplier, ratio of variance to mean
    - ``pascal``:
    - ``pascal``: (generalized) pascal-poisson distribution, a poisson stopped sum of negative binomial;
      exp_en gives the overall claim count. freq_a is the CV of the frequency distribution
      and freq_b is the number of claimants per claim (or claims per occurrence). Hence, the Poisson
      component has mean exp_en / freq_b and the number of claims per occurrence has mean freq_b. This
      parameterization may not be ideal(!).

    **Mixture** Frequency Types

    These distributions are G-mixed Poisson, so N | G ~ Poisson(n G). They are labelled by
    the name of the mixing distribution or the common name for the resulting frequency
    distribution. See Panjer and Willmot or JKK.

    In all cases freq_a is the CV of the mixing distribution which corresponds to the
    asympototic CV of the frequency distribution and of any aggregate when the severity has a variance.

    - ``gamma``: negative binomial, freq_a = cv of gamma distribution
    - ``delaporte``: shifted gamma, freq_a = cv of mixing disitribution, freq_b = proportion of
      certain claims = shift. freq_b must be between 0 and 1.
    - ``ig``: inverse gaussian, freq_a = cv of mixing distribution
    - ``sig``: shifted inverse gaussian, freq_a = cv of mixing disitribution, freq_b = proportion of
      certain claims = shift. freq_b must be between 0 and 1.
    - ``sichel``: generalized inverse gaussian mixing distribution, freq_a = cv of mixing distribution and
      freq_b = lambda value. The beta and mu parameters solved to match moments. Note lambda =
      -0.5 corresponds to inverse gaussian and 0.5 to reciprocal inverse gauusian. Other special
      cases are available.
    - ``sichel.gamma``: generalized inverse gaussian mixture where the parameters match the moments of a
      delaporte distribution with given freq_a and freq_b
    - ``sichel.ig``: generalized inverse gaussian mixture where the parameters match the moments of a
      shifted inverse gaussian distribution with given freq_a and freq_b. This parameterization
      has poor numerical stability and may fail.
    - ``beta``: beta mixing with freq_a = Cv where beta is supported on the interval [0, freq_b]. This
      method should be used carefully. It has poor numerical stability and can produce bizzare
      aggregates when the alpha or beta parameters are < 1 (so there is a mode at 0 or freq_b).

    Code proof for Neyman A::

        from aggregate import build, qd
        mean = 10
        eggs_per_cluster = 4
        neya = build(f'agg Neya {mean} claims dsev[1] neymana {eggs_per_cluster}')
        qd(neya)

        po = build(f'agg Po4 {eggs_per_cluster} claims dsev[1] poisson')
        po_pmf = po.density_df.query('p_total > 1e-13').p_total

        byhand = build(f'agg ByHand {mean / eggs_per_cluster} claims dsev {list(po_pmf.index)} {po_pmf.values} poisson')
        qd(byhand)

        df = pd.concat((neya.density_df.p_total, byhand.density_df.p_total), axis=1)
        df.columns = ['neya', 'byhand']
        df['err'] = df.neya - df.byhand
        assert df.err.abs().max() < 1e-5
        df.head(40)

    Code proof for Pascal::

        from aggregate import build, qd
        mean = 10
        claims_per_occ =1.24
        overall_cv = 1.255
        pascal = build(f'agg PascalEg {mean} claims dsev[1] pascal {overall_cv} {claims_per_occ}', log2=16)
        qd(pascal)

        c = (mean * overall_cv**2 - 1 - claims_per_occ) / claims_per_occ
        th = claims_per_occ * c
        a = 1 / c
        # from form of nb pgf identify r = a and beta = theta, mean is rb, var is rb(1+b)
        nb = build(f'agg NB {claims_per_occ} claims dsev[1] negbin {th + 1}', log2=16)
        nb_pmf = nb.density_df.query('p_total > 1e-13').p_total
        qd(nb)

        byhand = build(f'agg ByHand {mean / claims_per_occ} claims dsev {list(nb_pmf.index)} {nb_pmf.values} poisson', log2=16)
        qd(byhand)

        df = pd.concat((pascal.density_df.p_total, byhand.density_df.p_total), axis=1)
        df.columns = ['pascal', 'byhand']
        df['err'] = df.pascal - df.byhand
        assert df.err.abs().max() < 1e-5
        df.head(40)

    :param freq_name: name of the frequency distribution, poisson, geometric, etc.
    :param freq_a:
    :param freq_b:
    """

    # Registry of subclasses keyed by ``freq_name`` (registry key). Populated
    # by ``__init_subclass__`` as each ``Frequency<Kind>`` subclass is imported.
    _registry: dict = {}

    # Subclass contract — overridden on each ``Frequency<Kind>``:
    #   freq_name: registry key (e.g. 'poisson'). Empty on the base class.
    #   supports_zm: True iff the subclass defines ``prn_eq_0`` and supports
    #     zero modification.
    #   prn_eq_0: class-level default ``None``; ZM subclasses override with
    #     a method that returns P(N = 0 | mean = n).
    freq_name = ''
    supports_zm = False
    prn_eq_0 = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register concrete kinds — subclasses that don't set a
        # class-level ``freq_name`` (e.g. ``Aggregate`` during the transition
        # period) are not part of the registry.
        key = cls.__dict__.get('freq_name', '')
        if key:
            Frequency._registry[key] = cls

    def __new__(cls, freq_name=None, *args, **kwargs):
        """
        Factory dispatch: ``Frequency('poisson', ...)`` → ``FrequencyPoisson``.

        When called on a subclass directly or with ``freq_name=None``, no
        dispatch occurs. Direct calls on ``Frequency`` route to the
        registered subclass for the (normalized) ``freq_name``; if no
        subclass is registered, fall back to the base class so the legacy
        if/elif body in ``__init__`` can handle it during the migration.
        """
        if cls is not Frequency or freq_name is None:
            return object.__new__(cls)
        lookup = _normalize_freq_name(freq_name)
        subclass = cls._registry.get(lookup)
        if subclass is None:
            return object.__new__(cls)
        return object.__new__(subclass)

    def _solve_n_base(self, n):
        """
        Solve for ``n_base`` such that the ZM-adjusted mean equals ``n``.

        The ZM construction reweights an unmodified distribution of mean
        ``n_base`` so that its probability at 0 becomes ``freq_p0``. The
        new mean is ``(1 - freq_p0) n_base / (1 - prn_eq_0(n_base))``;
        invert that to recover ``n_base``.
        """
        p0 = self.freq_p0
        f = lambda x: (1 - p0) * x / (1 - self.prn_eq_0(x)) - n
        if p0 < self.prn_eq_0(n):
            # ZM has higher mean than the unmodified distribution: search left of n.
            n_base = brentq(f, a=0, b=n)
        else:
            # search right
            n_base = brentq(f, a=n, b=n / (1 - p0))
        self.unmodified_mean = n_base
        return n_base

    def _install_zm_wrappers(self):
        """
        Replace ``self.freq_moms`` and ``self.freq_pgf`` with ZM-adjusted
        versions. The wrapped form is a weighted average of the trivial
        (point-mass at 0) component and the original distribution.
        """
        orig_moms = self.freq_moms
        orig_pgf = self.freq_pgf
        freq_p0 = self.freq_p0

        @wraps(orig_moms)
        def wrapped_moms(n):
            n_base = self._solve_n_base(n)
            ans = np.array(orig_moms(n_base))
            return (1 - freq_p0) / (1 - self.prn_eq_0(n_base)) * ans

        @wraps(orig_pgf)
        def wrapped_pgf(n, z):
            n_base = self._solve_n_base(n)
            wt = (1 - freq_p0) / (1 - self.prn_eq_0(n_base))
            return (1 - wt) + wt * orig_pgf(n_base, z)

        self.freq_moms = wrapped_moms
        self.freq_pgf = wrapped_pgf

    def __init__(self, freq_name, freq_a, freq_b, freq_zm, freq_p0):
        """
        Creates the freq_pgf and moment function:

        * moment function(n) returns EN, EN^2, EN^3 when EN=n.
        * freq_pgf(n, z) is the freq_pgf evaluated at log(z) when EN=n

        :param freq_name: name of the frequency distribution, poisson, geometric, etc.
        :param freq_a:
        :param freq_b:
        :param freq_zm: freq_zm True if zero modified, default False
        :param freq_p0: modified p0, probability of zero claims
        """
        self.freq_name = freq_name
        self.freq_a = freq_a
        self.freq_b = freq_b
        self.freq_zm = freq_zm
        self.freq_p0 = freq_p0
        self.panjer_ab = None
        self.unmodified_mean = None

        # ``__new__`` dispatches ``Frequency(name, ...)`` to the matching
        # registered ``Frequency<Kind>``. Direct construction on the base
        # ``Frequency`` is only reachable if ``freq_name`` is unregistered.
        if type(self) is Frequency:
            raise ValueError(
                f'Inadmissible frequency type {freq_name!r}; '
                f'available: {sorted(Frequency._registry)}')

        self._build()
        if freq_zm:
            if not self.supports_zm:
                raise NotImplementedError(
                    f'Zero modification not implemented for {freq_name}')
            self._install_zm_wrappers()

    def __str__(self):
        """
        wrap default with name
        :return:
        """
        return f'Frequency object of type {self.freq_name}\n{super(Frequency, self).__repr__()}'

    @property
    def tail_description(self) -> str:
        """One line: this frequency family's right-tail class (count layer).

        E.g. ``poisson frequency, super-exponential count``. The count *support*
        depends on the exposure (the aggregate's ``n``), so a standalone
        frequency reports only its family class; the full count support appears
        in the aggregate's :attr:`~aggregate.distributions.Aggregate.tail_behavior_df`.
        """
        rung, _ = _tail.classify_frequency(self)
        return f'{self.freq_name} frequency, {_tail.tail_class_label(rung)} count'

    @cached_property
    def freq_df(self):
        """Count pmf vs mean-matched Poisson comparison table (on demand).

        Index ``n`` (count), columns ``p`` (this frequency's pmf) and
        ``po_p`` (the Poisson pmf with the same mean) -- an eyeball
        diagnostic for how far the count law sits from Poisson
        (over/under-dispersion, cluster fatness, renewal regularity).
        Computed on first access and cached.

        Only the empirical family (``dfreq`` / ``renewal``) materializes
        its own count distribution and mean; parametric kinds take the
        expected count from the Aggregate exposure at use time, so this
        base implementation raises ``ValueError`` -- use
        :meth:`Aggregate.freq_pmf` for a parametric count pmf.
        """
        raise ValueError(
            f'freq_df: a parametric {self.freq_name!r} frequency has no '
            f'fixed expected claim count (n comes from the Aggregate '
            f'exposure); freq_df is available on dfreq/renewal (empirical) '
            f'frequencies -- use Aggregate.freq_pmf(log2) here')


# ---------------------------------------------------------------------------
# Concrete Frequency<Kind> subclasses. Each declares its registry key as a
# class-level ``freq_name`` and implements ``_build``, ``freq_moms``,
# ``freq_pgf``, and (for kinds that support ZM) ``prn_eq_0``.
# ---------------------------------------------------------------------------


class FrequencyPoisson(Frequency):
    """
    Poisson(n) frequency. Single-parameter: mean ``n``, variance ``n``.

    ``freq_a`` is unused. PGF :math:`G_N(z) = e^{n(z - 1)}`. Supports
    zero modification via the shared ZM machinery.
    """

    freq_name = 'poisson'
    supports_zm = True

    def _build(self):
        # No precomputation; freq_a is unused for pure Poisson.
        return None

    def prn_eq_0(self, n):
        return np.exp(-n)

    def freq_moms(self, n):
        freq_2 = n * (1 + n)
        freq_3 = n * (1 + n * (3 + n))
        self.panjer_ab = (0., n)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        return np.exp(n * (z - 1))


class FrequencyFixed(Frequency):
    """
    Degenerate frequency: ``N = n`` with probability 1. No parameters.
    """

    freq_name = 'fixed'

    def _build(self):
        return None

    def freq_moms(self, n):
        freq_2 = n ** 2
        freq_3 = n * freq_2
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        return z ** n


class FrequencyBernoulli(Frequency):
    """
    Bernoulli frequency. ``n`` is interpreted as the probability of a single
    claim (so ``n < 1``); :math:`E(N^k) = n` for all :math:`k`.
    """

    freq_name = 'bernoulli'

    def _build(self):
        return None

    def freq_moms(self, n):
        return n, n, n

    def freq_pgf(self, n, z):
        return z * n + np.ones_like(z) * (1 - n)


class FrequencyBinomial(Frequency):
    """
    Binomial(N, p) frequency with overall mean ``n`` and per-trial success
    probability ``p = freq_a``. The trial count is :math:`N = n / p`.
    Supports zero modification.
    """

    freq_name = 'binomial'
    supports_zm = True

    def _build(self):
        # ``freq_a`` carries the per-trial probability ``p``; trial count
        # is derived at evaluation time from the requested mean.
        return None

    def prn_eq_0(self, n):
        p = self.freq_a
        N = n / p
        return (1 - p) ** N

    def freq_moms(self, n):
        p = self.freq_a
        N = n / p
        freq_1 = N * p
        freq_2 = N * p * (1 - p + N * p)
        freq_3 = N * p * (1 + p * (N - 1) * (3 + p * (N - 2)))
        self.panjer_ab = (-p / (1 - p), (N + 1) * p / (1 - p))
        return freq_1, freq_2, freq_3

    def freq_pgf(self, n, z):
        p = self.freq_a
        N = n / p
        return (z * p + np.ones_like(z) * (1 - p)) ** N


class FrequencyNegbin(Frequency):
    """
    Negative binomial with ``freq_a`` interpreted as the variance multiplier
    (variance / mean). Parameterized via ``r, β`` with ``β = freq_a - 1``
    and ``r = n / β``; mean ``rβ``, variance ``rβ(1+β)``. Supports zero
    modification.
    """

    freq_name = 'negbin'
    supports_zm = True

    def _build(self):
        self._beta = self.freq_a - 1

    def prn_eq_0(self, n):
        beta = self._beta
        r = n / beta
        return (1 + beta) ** -r

    def freq_moms(self, n):
        beta = self._beta
        r = n / beta
        freq_2 = n * (1 + beta * (1 + r))
        freq_3 = r * beta * (1 + beta * (1 + r) * (3 + beta * (2 + r)))
        self.panjer_ab = (beta / (1 + beta), (r - 1) * beta / (1 + beta))
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        beta = self._beta
        r = n / beta
        return (1 - beta * (z - 1)) ** -r


class FrequencyGeometric(Frequency):
    """
    Geometric distribution supported on 0, 1, 2, ... with mean ``n``, hence
    success probability ``p = 1 / (n + 1)``. Supports zero modification.
    """

    freq_name = 'geometric'
    supports_zm = True

    def _build(self):
        return None

    def prn_eq_0(self, n):
        return 1 / (n + 1)

    def freq_moms(self, n):
        p = 1 / (n + 1)
        freq_2 = (2 - p) * (1 - p) / p ** 2
        freq_3 = (1 - p) * (6 + (p - 6) * p) / p ** 3
        self.panjer_ab = (n / (1 + n), 0.)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        p = 1 / (n + 1)
        return p / (1 - (1 - p) * z)


class FrequencyLogarithmic(Frequency):
    """
    Logarithmic series (``logser``) supported on 1, 2, 3, ... with mean
    ``n``; the parameter ``θ`` is solved numerically by
    :func:`_logarithmic_theta`. Supports zero modification (with
    ``prn_eq_0 = 0`` for the unmodified form, so ZM only adds mass at zero).
    """

    freq_name = 'logarithmic'
    supports_zm = True

    def _build(self):
        return None

    def prn_eq_0(self, n):
        return 0.

    def freq_moms(self, n):
        theta = _logarithmic_theta(n)
        a_logser = -1 / np.log(1 - theta)
        freq_2 = a_logser * theta / (1 - theta) ** 2
        freq_3 = a_logser * theta * (1 + theta) / (1 - theta) ** 3
        self.panjer_ab = (theta, -theta)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        theta = _logarithmic_theta(n)
        return np.log(1 - theta * z) / np.log(1 - theta)


class FrequencyNeymanA(Frequency):
    """
    Neyman A: Poisson stopped sum of Poisson. ``freq_a`` is the mean number
    of outcomes per cluster (``m2``); the overall mean is ``n = m1 * m2``.
    Aliases ``'neyman'`` and ``'neymanA'`` route here.
    """

    freq_name = 'neymana'

    def _build(self):
        self._m2 = self.freq_a

    def freq_moms(self, n):
        m2 = self._m2
        freq_2 = n * ((1 + m2) + n)
        freq_3 = n * ((1 + m2 * (3 + m2)) + 3 * freq_2 - 2 * n ** 2)
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        m2 = self._m2
        m1 = n / m2
        return np.exp(m1 * (np.exp(m2 * (z - 1)) - 1))


class FrequencyPascal(Frequency):
    """
    Generalized Poisson-Pascal: Poisson stopped sum of negative binomials.
    ``freq_a`` is the overall CV ``ν``; ``freq_b`` is the mean claimants per
    claim ``κ``. The Poisson component has mean ``n / κ``.
    """

    freq_name = 'pascal'

    def _build(self):
        self._nu = self.freq_a
        self._kappa = self.freq_b

    def freq_moms(self, n):
        nu = self._nu
        kappa = self._kappa
        c = (n * nu ** 2 - 1 - kappa) / kappa
        lam = n / kappa
        g = kappa * lam * (
            2 * c ** 2 * kappa ** 2 + 3 * c * kappa ** 2 * lam + 3 * c * kappa ** 2 + 3 * c * kappa
            + kappa ** 2 * lam ** 2 + 3 * kappa ** 2 * lam + kappa ** 2 + 3 * kappa * lam + 3 * kappa + 1)
        return n, n * (kappa * (1 + c + lam) + 1), g

    def freq_pgf(self, n, z):
        nu = self._nu
        kappa = self._kappa
        c = (n * nu ** 2 - 1 - kappa) / kappa
        a = 1 / c
        theta = kappa * c
        lam = n / kappa
        return np.exp(lam * ((1 - theta * (z - 1)) ** -a - 1))


class FrequencyEmpirical(Frequency):
    """
    Empirical (user-supplied) discrete frequency. ``freq_a`` is the array of
    outcomes, ``freq_b`` the array of probability masses; both are
    validated and possibly summarized via ``validate_discrete_distribution``.
    Moments are independent of the requested mean ``n`` (which is ignored).
    """

    freq_name = 'empirical'

    def _build(self):
        self.freq_a, self.freq_b = validate_discrete_distribution(
            self.freq_a, self.freq_b)

    def freq_moms(self, n):
        en = np.sum(self.freq_a * self.freq_b)
        en2 = np.sum(self.freq_a ** 2 * self.freq_b)
        en3 = np.sum(self.freq_a ** 3 * self.freq_b)
        return en, en2, en3

    def freq_pgf(self, n, z):
        # Horner / sorted-gap square-and-multiply dispatch; fractional
        # outcomes fall back to the legacy matrix expression. See
        # ``evaluate_pgf_polynomial`` ([Empirical-PGF-Horner-Dispatch]).
        return evaluate_pgf_polynomial(self.freq_a, self.freq_b, z)

    @cached_property
    def freq_df(self):
        """Count pmf vs mean-matched Poisson comparison table (on demand).

        Index ``n = 0..max(support)`` (holes in a sparse support carry
        ``p = 0``); columns ``p`` (this frequency's pmf) and ``po_p``
        (the Poisson pmf with the same mean). Computed on first access
        and cached. A small-count eyeball diagnostic: guards require a
        non-negative integer support and mean <= 1000.
        """
        a = np.asarray(self.freq_a, dtype=float)
        b = np.asarray(self.freq_b, dtype=float)
        mean = float(a @ b)
        if mean > 1000:
            raise ValueError(
                f'freq_df: mean frequency {mean:.6g} > 1000 -- the '
                f'comparison table is a small-count diagnostic')
        k = np.rint(a).astype(int)
        if np.any(np.abs(a - k) > 1e-9) or np.any(k < 0):
            raise ValueError(
                'freq_df: the Poisson comparison needs a non-negative '
                'integer count support')
        p = np.zeros(k.max() + 1)
        np.add.at(p, k, b)
        df = pd.DataFrame({'p': p,
                           'po_p': ss.poisson.pmf(np.arange(len(p)), mean)})
        df.index.name = 'n'
        return df


class FrequencyRenewal(FrequencyEmpirical):
    """
    Sparre-Andersen renewal count: the claim count is ``N(T)``, the number
    of renewals of an iid waiting-time law ``W`` over ``years = T``.

    Constructed DIRECTLY (never through the ``Frequency(name, ...)``
    factory) with the wait-law payload; ``_build`` computes the count pmf
    via :func:`aggregate._renewal.wait_count_pmf` and then the object *is*
    an ordinary empirical frequency: ``freq_a = 0..kmax``, ``freq_b = pN``
    is exactly the ``dfreq [k...] [p_k...]`` representation, and every
    downstream consumer (Horner pgf, moments, the ``exp_en = -1`` count
    derivation, count support, ``freq_pmf``) runs the inherited empirical
    logic. The renewal computation only ever *produces* that vector pair.

    Parameters
    ----------
    wait_components : list of (Severity, lb, ub, conditional)
        The wait law mixture components (see
        :func:`aggregate._renewal.wait_count_pmf`).
    wait_weights : array-like
        Mixture weights; may sum to < 1 (the shortfall is defect mass --
        a terminating renewal process).
    years : float
        The horizon ``T``.

    Notes
    -----
    Diagnostics stored for repr / drill-down: ``wait_bs``, ``wait_log2``,
    ``wait_lattice``, ``wait_snapped`` (layered cap atom phase-aligned to
    the lattice, closed readout), ``wait_p0`` (zero-wait cluster mass),
    ``wait_defect`` (terminating mass), ``kmax``, ``years``, and
    ``_renewal_bs_df`` (the
    grid-sizing constraint table, mirroring the aggregate ``_bs_window_df``
    idiom). ``convergence_check()`` is the explicit-opt-in Richardson
    diagnostic. Zero modification is meaningless for a renewal count
    (``supports_zm = False``, inherited).
    """

    freq_name = 'renewal'

    def __init__(self, wait_components, wait_weights, years):
        # stash the payload BEFORE super().__init__, which calls _build
        self.wait_components = wait_components
        self.wait_weights = wait_weights
        self.years = years
        self.wait_bs = None
        self.wait_log2 = None
        self.wait_lattice = None
        self.wait_snapped = None
        self.wait_p0 = None
        self.wait_defect = None
        self.kmax = None
        self._renewal_bs_df = None
        super().__init__('renewal', None, None, False, np.nan)

    def _build(self):
        from ._renewal import wait_count_pmf
        k, pN, info = wait_count_pmf(self.wait_components,
                                     self.wait_weights, self.years)
        self.freq_a = k.astype(float)
        self.freq_b = pN
        self.wait_bs = info['bs']
        self.wait_log2 = info['log2']
        self.wait_lattice = info['lattice']
        self.wait_snapped = info['snapped']
        self.wait_p0 = info['p0']
        self.wait_defect = info['defect']
        self.kmax = info['kmax']
        self._renewal_bs_df = info['bs_df']
        # inherited empirical validation (sorts, checks mass, dedups)
        super()._build()

    def convergence_check(self):
        """Richardson-style grid diagnostic: recompute the count pmf at h/2.

        Explicit opt-in (never run automatically). Returns ``max |delta
        p_k|`` between the production count pmf and one recomputed on a
        grid with half the bucket size (one extra log2). O(h^2)
        convergence means the reported delta is ~4x the remaining error
        of the *refined* pmf. Meaningless (and skipped -- returns 0.0) on
        an exact lattice, which has no discretization error. On a snapped
        grid (layered cap atom phase-aligned to the lattice) the closed
        readout is kept at h/2 -- bs still divides the atom step -- and
        the continuous part converges at O(h), so the delta is ~2x the
        remaining error.
        """
        from ._renewal import wait_count_pmf
        if self.wait_lattice:
            return 0.0
        k2, pN2, _ = wait_count_pmf(
            self.wait_components, self.wait_weights, self.years,
            grid=(self.wait_bs / 2, self.wait_log2 + 1,
                  bool(self.wait_snapped)))
        n = min(len(pN2), len(self.freq_b))
        return float(np.abs(pN2[:n] - self.freq_b[:n]).max())


class _FrequencyMixedPoisson(Frequency):
    """
    Shared scaffolding for G-mixed Poisson kinds: :math:`N \\mid G \\sim
    \\text{Poisson}(nG)` where ``G`` is a non-negative mixing distribution
    with mean 1 and CV ``ν = freq_a``. Subclasses compute ``g = E[G^3]``
    in ``_build`` (stored as ``self._g``) and supply their own
    ``freq_pgf``.

    All G-mixed Poissons share the same factorial-moment formulas because
    ``EN^k`` reduces to the non-central moments of ``G`` scaled by ``n``.
    """

    def freq_moms(self, n):
        c = self._c
        g = self._g
        freq_2 = n * (1 + (1 + c) * n)
        freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
        return n, freq_2, freq_3


class FrequencyGammaMixed(_FrequencyMixedPoisson):
    """
    Gamma-mixed Poisson (= negative binomial). ``freq_a`` is the CV ``ν`` of
    the gamma mixing distribution; the resulting frequency has variance
    multiplier ``1 + n ν²``.
    """

    freq_name = 'gamma'

    def _build(self):
        nu = self.freq_a
        c = nu * nu
        self._c = c
        self._a = 1 / c
        self._theta = c
        self._g = 1 + 3 * c + 2 * c * c

    def freq_pgf(self, n, z):
        return (1 - self._theta * n * (z - 1)) ** -self._a


class FrequencyDelaporteMixed(_FrequencyMixedPoisson):
    """
    Delaporte-mixed Poisson: shifted gamma with a proportion of certain
    claims ``f = freq_b`` (must lie in ``[0, 1]``). ``freq_a`` is the CV of
    the mixing distribution.
    """

    freq_name = 'delaporte'

    def _build(self):
        nu = self.freq_a
        c = nu * nu
        f = self.freq_b
        a = (1 - f) ** 2 / c
        self._c = c
        self._f = f
        self._a = a
        self._theta = (1 - f) / a
        self._g = 2 * nu ** 4 / (1 - f) + 3 * c + 1

    def freq_pgf(self, n, z):
        f = self._f
        theta = self._theta
        a = self._a
        return np.exp(f * n * (z - 1)) * (1 - theta * n * (z - 1)) ** -a


class FrequencyIGMixed(_FrequencyMixedPoisson):
    """
    Inverse-gaussian-mixed Poisson. ``freq_a`` is the CV of the mixing
    distribution.
    """

    freq_name = 'ig'

    def _build(self):
        nu = self.freq_a
        c = nu ** 2
        mu = c
        lam = 1 / mu
        gamma_skew = 3 * np.sqrt(mu)
        self._c = c
        self._mu = mu
        self._lam = lam
        self._g = gamma_skew * nu ** 3 + 3 * c + 1

    def freq_pgf(self, n, z):
        mu = self._mu
        lam = self._lam
        return np.exp(1 / mu * (1 - np.sqrt(1 - 2 * mu ** 2 * lam * n * (z - 1))))


class FrequencySIGMixed(_FrequencyMixedPoisson):
    """
    Shifted inverse-gaussian-mixed Poisson. ``freq_a`` is the CV of the
    mixing distribution; ``freq_b`` is the proportion of certain claims.
    """

    freq_name = 'sig'

    def _build(self):
        nu = self.freq_a
        f = self.freq_b
        c = nu * nu
        mu = c / (1 - f) ** 2
        lam = (1 - f) / mu
        gamma_skew = 3 * np.sqrt(mu)
        self._c = c
        self._f = f
        self._mu = mu
        self._lam = lam
        self._g = gamma_skew * nu ** 3 + 3 * c + 1

    def freq_pgf(self, n, z):
        f = self._f
        mu = self._mu
        lam = self._lam
        return (np.exp(f * n * (z - 1))
                * np.exp(1 / mu * (1 - np.sqrt(1 - 2 * mu ** 2 * lam * n * (z - 1)))))


class FrequencyBetaMixed(_FrequencyMixedPoisson):
    """
    Beta-mixed Poisson over support :math:`[0, r]` (``r = freq_b > 1``) with
    mixing CV ``ν = freq_a``. Numerically unstable when the implied
    alpha/beta parameters approach 1.
    """

    freq_name = 'beta'

    def _build(self):
        nu = self.freq_a
        r = self.freq_b
        assert r > 1, f'beta-mixed Poisson requires r > 1, got {r}'
        self._c = nu * nu
        self._r = r
        # ``g`` depends on ``n`` so is computed lazily in ``freq_moms``.

    def freq_moms(self, n):
        c = self._c
        r = self._r
        b = (r - n * (1 + c)) * (r - n) / (c * n * r)
        a = n / (r - n) * b
        g = r ** 3 * np.exp(
            gammaln(a + b) + gammaln(a + 3) - gammaln(a + b + 3) - gammaln(a))
        freq_2 = n * (1 + (1 + c) * n)
        freq_3 = n * (1 + n * (3 * (1 + c) + n * g))
        return n, freq_2, freq_3

    def freq_pgf(self, n, z):
        c = self._c
        r = self._r
        b = (r - n * (1 + c)) * (r - n) / (c * n * r)
        a = (r - n * (1 + c)) / (c * r)
        return hyp1f1(a, a + b, r * (z - 1))


class _FrequencySichelBase(_FrequencyMixedPoisson):
    """
    Shared body for the Sichel family (generalized inverse-gaussian mixing).
    Subclasses provide a ``_calibrate`` method returning ``(mu, beta, lam)``
    via Broyden / Newton-Krylov; the base class wires the resulting
    parameters into ``freq_pgf``.
    """

    def _build(self):
        nu = self.freq_a
        self._nu = nu
        self._c = nu * nu
        mu, beta, lam = self._calibrate()
        self._mu = mu
        self._beta = beta
        self._lam = lam
        self._g = mu ** 2 * kv(lam + 2, mu / beta) / kv(lam, mu / beta)

    def freq_pgf(self, n, z):
        mu = self._mu
        beta = self._beta
        lam = self._lam
        kernel = n * (z - 1)
        inner = np.sqrt(1 - 2 * beta * kernel)
        return inner ** (-lam) * kv(lam, mu * inner / beta) / kv(lam, mu / beta)


class FrequencySichel(_FrequencySichelBase):
    """
    Pure Sichel: generalized inverse-gaussian mixing with shape parameter
    ``λ = freq_b``. The other GIG parameters ``μ, β`` are calibrated by
    Broyden to match mean = 1 and CV = ``freq_a``. Special cases include
    ``λ = -0.5`` (inverse gaussian) and ``λ = 0.5`` (reciprocal IG).
    """

    freq_name = 'sichel'

    def _calibrate(self):
        nu = self.freq_a
        lam = self.freq_b
        target = np.array([1, nu])
        mu = 1
        beta = nu ** 2

        def f(arr_in):
            mu_, beta_ = arr_in
            mu_ = np.exp(mu_)
            beta_ = np.exp(beta_)
            ex1, ex2 = np.array(
                [mu_ ** r * kv(lam + r, mu_ / beta_) / kv(lam, mu_ / beta_)
                 for r in (1, 2)])
            sd = np.sqrt(ex2 - ex1 * ex1)
            return np.array([ex1, sd]) - target

        try:
            params = broyden2(f, (np.log(mu), np.log(beta)),
                              verbose=False, iter=10000, f_rtol=1e-11)
        except NoConvergence as e:
            logger.error('Sichel calibration: Broyden did not converge: %s', e)
            raise

        logger.debug('sichel params from Broyden %s', params)
        mu_, beta_ = params
        return np.exp(mu_), np.exp(beta_), lam


class _FrequencySichelMatched(_FrequencySichelBase):
    """
    Shared body for Sichel kinds calibrated by matching the first three
    moments of another distribution (delaporte or shifted IG). Subclasses
    supply ``_match_target(nu, f)`` returning the target moment vector.
    """

    def _calibrate(self):
        nu = self.freq_a
        f = self.freq_b
        lam = -0.5
        mu = 1
        beta = nu ** 2
        target = self._match_target(nu, f)

        def fn(arr_in):
            mu_, beta_, lam_ = arr_in
            mu_ = np.exp(mu_)
            beta_ = np.exp(beta_)
            ex1, ex2, ex3 = np.array(
                [mu_ ** r * kv(lam_ + r, mu_ / beta_) / kv(lam_, mu_ / beta_)
                 for r in (1, 2, 3)])
            sd = np.sqrt(ex2 - ex1 * ex1)
            skew = (ex3 - 3 * ex2 * ex1 + 2 * ex1 ** 3) / (sd ** 3)
            return np.array([ex1, sd, skew]) - target

        try:
            params = broyden2(fn, (np.log(mu), np.log(beta), lam),
                              verbose=False, iter=10000, f_rtol=1e-11)
            if np.linalg.norm(params) > 20:
                # Fall back to Newton-Krylov on suspiciously large solutions.
                params1 = newton_krylov(
                    fn, (np.log(1.0), np.log(nu ** 2), -0.5),
                    verbose=False, iter=10000, f_rtol=1e-11)
                logger.warning(
                    f'{self.freq_name}: Broyden gave large result {params}; '
                    f'Newton-Krylov {params1}')
                if np.linalg.norm(params) > np.linalg.norm(params1):
                    params = params1
                    logger.warning('%s: using Newton-Krylov', self.freq_name)
        except NoConvergence as e:
            logger.error('%s calibration: Broyden did not converge: %s', self.freq_name, e)
            raise

        logger.debug('%s params from Broyden %s', self.freq_name, params)
        mu_, beta_, lam_ = params
        return np.exp(mu_), np.exp(beta_), lam_


class FrequencySichelGamma(_FrequencySichelMatched):
    """
    Sichel calibrated to delaporte moments: ``G = f + G'`` with
    ``E(G') = 1 - f``, matching SD and skewness of the corresponding
    delaporte distribution.
    """

    freq_name = 'sichel.gamma'

    def _match_target(self, nu, f):
        return np.array([1, nu, 2 * nu / (1 - f)])


class FrequencySichelIG(_FrequencySichelMatched):
    """
    Sichel calibrated to shifted-inverse-gaussian moments. Numerically
    fragile parameterization; may fail to converge for some inputs.
    """

    freq_name = 'sichel.ig'

    def _match_target(self, nu, f):
        return np.array([1, nu, 3.0 * nu / (1 - f)])
