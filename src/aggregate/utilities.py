from collections import namedtuple
import inspect
import itertools
import logging
from numbers import Number

import numpy as np
import pandas as pd
import re
import scipy.fft as sft

# IPython is deliberately NOT imported at module scope: `from IPython.display
# import ...` costs ~1s and utilities.py sits on the `import aggregate` path
# (distributions/portfolio/everything import it). The only consumer left is
# `agg_help`, so it is imported lazily inside that function instead. (Program
# pretty-printing/colorization moved to aggregate.decl_writer at 1.0.0a53.)

from .constants import Validation


logger = logging.getLogger(__name__)

__all__ = [
    'ft', 'ift',
    'subsets',
    'remove_fuzz',
    'round_bucket',
    'nice_multiple',
    'qd', 'mv',
    'make_var_tvar', 'kaplan_meier', 'kaplan_meier_np',
    'agg_help', 'explain_validation', 'introspect',
    'silence_warnings',
]


def silence_warnings(category=Warning, message='', module=''):
    """Suppress matching warnings in the current process.

    Convenience for notebook / REPL users who'd rather not see numpy / scipy
    chatter while exploring. With no arguments it silences *all* warnings;
    narrow it by passing a ``category``, a ``message`` regex, or a ``module``
    regex. For example, the benign boundary-evaluation noise from the
    distortion g-functions is all :class:`RuntimeWarning`::

        from aggregate.utilities import silence_warnings
        silence_warnings(RuntimeWarning)                  # just those
        silence_warnings(message='divide by zero')        # even narrower
        silence_warnings()                                # everything

    Parameters
    ----------
    category : type[Warning], default :class:`Warning`
        Warning class to ignore. The default :class:`Warning` matches every
        category; pass e.g. ``RuntimeWarning`` or ``DeprecationWarning`` to
        scope it.
    message : str, default ''
        Regex matched against the *start* of the warning text; '' matches all.
    module : str, default ''
        Regex matched against the issuing module's ``__name__``; '' matches all.

    Notes
    -----
    Appends an ``ignore`` rule via :func:`warnings.filterwarnings`. Unlike the
    blunter ``warnings.simplefilter('ignore')`` it does **not** discard
    existing filters. The effect is process-wide and is an explicit
    end-user-only convenience -- never call it from library code.
    """
    import warnings
    warnings.filterwarnings('ignore', message=message, category=category, module=module)


def ft(z, padding):
    """
    fft with padding
    padding = n makes vector 2^n as long
    n=1 doubles (default)
    n=2 quadruples

    :param z:
    :param padding: = 1 doubles
    :return:
    """
    locft = sft.rfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    # valeus per https://stackoverflow.com/questions/71706387/finding-fft-gives-keyerror-aligned-pandas
    zt = z
    if type(zt) != np.ndarray:
        zt = zt.to_numpy()
    # padding handled by the ft routine
    # temp = np.hstack((z, np.zeros_like(z)))
    return locft(zt, len(z) << padding)


def ift(z, padding):
    """
    ift that strips out padding

    :param z:
    :param padding:
    :return:
    """
    locift = sft.irfft
    if z.shape != (len(z),):
        raise ValueError('ERROR wrong shape passed into ft: ' + str(z.shape))
    if type(z) != np.ndarray:
        temp = locift(z.to_numpy())
    else:
        temp = locift(z)
    # unpad
    if padding != 0:
        temp = temp[0:len(temp) >> padding]
    return temp


def remove_fuzz(data, eps=None):
    """Zero entries with ``|x| < eps`` (machine-epsilon FFT/round-off noise).

    Vectorized replacement for the scattered de-fuzz idioms. Accepts an ndarray
    or a DataFrame; for a DataFrame only the float64 columns are touched and a
    new frame is returned (callers that need in-place semantics assign the
    result back). Two-sided: large negatives are preserved (unlike a one-sided
    ``x < eps`` clip), so it is correct on signed / P&L densities.

    Parameters
    ----------
    data : numpy.ndarray | pandas.DataFrame
        Array or frame to clean. The input is never mutated.
    eps : float, optional
        Threshold; defaults to ``np.finfo(float).eps`` (~2.22e-16).

    Returns
    -------
    numpy.ndarray | pandas.DataFrame
        Same type as ``data``, a copy with sub-eps entries set to ``0.0``.

    Notes
    -----
    The raw inverse-FFT density carries sub-machine-epsilon fuzz (tiny +/-
    values) in essentially every bucket. In a plain mass sum this cancels, but
    moment calculations weight each bucket by ``x**k``, so far-tail fuzz at
    large ``x`` is amplified and corrupts the empirical skew. Zeroing
    ``|x| < eps`` is safe and lossless: the exact aggregate has no genuine
    density below machine epsilon.

    The one-sided ``Frequency.pmf`` clip and the plot-cosmetic ``1e-15`` clip in
    the reinsurance occurrence plot are deliberate carve-outs that do NOT route
    through this helper -- they have different (one-sided / looser) semantics.
    """
    if eps is None:
        eps = np.finfo(float).eps
    if isinstance(data, pd.DataFrame):
        out = data.copy()
        float_cols = out.select_dtypes(include=['float64']).columns
        out[float_cols] = out[float_cols].mask(out[float_cols].abs() < eps, 0.0)
        return out
    return np.where(np.abs(data) < eps, 0.0, data)


#: Mantissa ladder for ``bs >= 1``: the "nice" bucket sizes within each decade.
#: Consecutive ratios (1, 2, 4, 5, 8, 10) are all <= 2, so rounding *up* to the
#: next rung overshoots the requested ``bs`` by less than 2x -- no 2.5x jumps
#: (the old 2 -> 5 / 20 -> 50 gaps). ``10`` rolls into the next decade's ``1``.
_BUCKET_LADDER = (1, 2, 4, 5, 8, 10)


def round_bucket(bs):
    """Round ``bs`` *up* to a "nice" bucket size.

    The grid step ``bs`` is chosen to **cover** the support, so this rounds up
    (never down) to the next nice value. Two regimes, both with <= 2x rung gaps
    so the overshoot is bounded below 2x:

    - ``bs >= 1`` -- the smallest member of the decade ladder
      ``{1, 2, 4, 5, 8} * 10**k`` that is ``>= bs`` (so 3.4 -> 4, not 5; 5.5 ->
      8; 9 -> 10). See :data:`_BUCKET_LADDER`.
    - ``bs < 1`` -- the smallest power of two ``>= bs`` (``..., 1/4, 1/2, 1``).
      Kept binary-exact: a sub-unit ``bs`` divides the FFT grid, and powers of
      two are exact in floating point (0.2 / 0.4 / 0.8 are not), while still
      honouring the <= 2x rule.

    There is deliberately no "round to nearest" mode. Rounding *down* would
    leave ``bs`` failing to cover the support, and it cannot tighten a grid
    anyway: the grid length is a power of two, so a power-of-two ``bs`` rounded
    down just forces a larger ``log2`` to recover coverage (more memory) or
    clips the tail.

    Parameters
    ----------
    bs : float
        Raw (un-rounded) bucket size; must be finite and non-zero.

    Returns
    -------
    float
        The rounded bucket size, ``>= bs``.

    Examples
    --------
    Test cases: ::

        test_cases = [1, 1.1, 2, 2.5, 4, 5, 5.5, 8.7, 9.9, 10, 13,
                      15, 20, 50, 100, 99, 101, 200, 250, 400, 457,
                        500, 750, 1000, 2412, 12323, 57000, 119000,
                        1e6, 1e9, 1e12, 1e15, 1e18, 1e21]
        for i in test_cases:
            print(i, round_bucket(i))
        for i in test_cases:
            print(1/i, round_bucket(1/i))
    """
    if bs == 0 or np.isinf(bs):
        raise ValueError(f'Inadmissible value passed to round_bucket, {bs}')

    if bs >= 1:
        # smallest {1,2,4,5,8}*10**k >= bs (round up within the decade)
        exp = int(np.floor(np.log10(bs)))
        base = 10.0 ** exp
        m = bs / base                       # mantissa in [1, 10)
        for rung in _BUCKET_LADDER:
            if m <= rung * (1 + 1e-9):       # tolerance for fp dust on exact rungs
                return float(rung * base)
        return 10.0 * base                   # m ~ 10 -> next decade (defensive)

    # bs < 1: smallest power of two >= bs (binary-exact small buckets). 1/bs > 1,
    # so floor it and take the largest power of two <= that via int.bit_length;
    # the reciprocal is the smallest power of two >= bs.
    n = int(1.0 / bs)
    return 1.0 / (1 << (n.bit_length() - 1))


# Logger configuration is controlled by the user of the package, not the package itself.


def subsets(x):
    """
    all non empty subsets of x, an interable
    """
    return list(itertools.chain.from_iterable(
        itertools.combinations(x, n) for n in range(len(x) + 1)))[1:]


# new graphics methods
def nice_multiple(mx):
    """
    Suggest a nice multiple for an axis with scale 0 to mx. Used by the MultipleLocator in discrete plots,
    where you want an integer multiple. Return 0 to let matplotlib figure the answer. Real issue is stopping
    multiples like 2.5.

    :param mx:
    :return:
    """
    m = mx / 6
    if m < 0:
        return 0

    m = mx // 6
    m = {3: 2, 4: 5, 6: 5, 7: 5, 8: 10, 9: 10}.get(m, m)
    if m < 10:
        return m

    # punt back to mpl for larger values
    return 0



def qd(*argv, accuracy=3, align=True, trim=True, ff=None, **kwargs):
    """
    Endless quest for a robust display format!

    Quick display (qd) a list of objects.
    Dataframes handled in text with reasonable defaults.
    For use in documentation.

    :param: argv: list of objects to print
    :param: accuracy: number of decimal places to display
    :param: align: legacy alignment flag (currently no-op; was used by removed engineering formatter)
    :param: trim: legacy trailing-zero trim flag (currently no-op)
    :param: ff: if not None, use this function to format floats, or 'basic', or 'binary'
    :kwargs: passed to pd.DataFrame.to_string for dataframes only. e.g., pass dict of formatters by column.

    """
    from .distributions import Aggregate
    from .portfolio import Portfolio
    if ff is None:
        ff = lambda x: f'{x:.5g}'
    elif ff == 'basic':
        ff = lambda x: f'{x:.1%}' if x < 1 else f'{x:12,.0f}'
    elif ff == 'int_ratio':
        def format_function(x):
            ir = np.round(x, 13).as_integer_ratio()
            return f'{int(x)}' if x in [0, 1] else f'  {ir[0]}/{ir[1]}'

        ff = format_function
    # split output
    for x in argv:
        if isinstance(x, (Aggregate, Portfolio)):
            # Drop the noisy CV-error column (``Err CV`` for the legacy
            # validation view; ``Change CV`` under reinsurance) for the
            # compact ``qd`` rendering; keep everything else.
            cols = x.describe.columns
            drop = [c for c in ('Err CV', 'Change CV') if c in cols]
            if drop:
                qd(x.describe.drop(columns=drop).fillna(''), accuracy=accuracy, **kwargs)
            else:
                # object not updated
                qd(x.describe.fillna(''), accuracy=accuracy, **kwargs)
            bss = 'na' if x.bs == 0 else (f'{x.bs:.0f}' if x.bs >= 1 else f'1/{1/x.bs:.0f}')
            vr = x.validation_explanation
            print(f'log2 = {x.log2}, bandwidth = {bss}, validation: {vr}.')
        elif isinstance(x, pd.DataFrame):
            # 100 line width matches rtd html format
            args = {'line_width': 100,
                    'max_cols': 35,
                    'max_rows': 25,
                    'float_format': ff,
                    # needs to be larger for text output
                    # 'max_colwidth': 10,
                    'sparsify': True,
                    'justify': None
                    }
            args.update(kwargs)
            print()
            print(x.to_string(**args))
            # print(x.to_string(formatters={c: f for c in x.columns}))
        elif isinstance(x, pd.Series):
            args = {'max_rows': 25,
                    'float_format': ff,
                    'name': True
                    }
            args.update(kwargs)
            print()
            print(x.to_string(**args))
        elif isinstance(x, int):
            print(x)
        elif isinstance(x, Number):
            print(ff(x))
        else:
            print(x)


def mv(x, y=None):
    """
    Nice display of mean and variance for Aggregate or Portfolios or
    entered values.

    R style function, no return value.

    :param x: Aggregate or Portfolio or float
    :param y: float, if x is a float
    :return: None
    """
    from .distributions import Aggregate
    from .portfolio import Portfolio
    if y is None and isinstance(x, (Aggregate, Portfolio)):
        print(f'mean     = {x.agg_m:.6g}')
        print(f'variance = {x.agg_var:.7g}')
        print(f'std dev  = {x.agg_sd:.6g}')
    else:
        print(f'mean     = {x:.6g}')
        print(f'variance = {y:.7g}')
        print(f'std dev  = {y**.5:.6g}')


def make_var_tvar(ser):
    """
    Make var (lower quantile), upper quantile, and tvar functions from a ``pd.Series`` ``ser``, which
    has index given by losses and p_total values.

    ``ser`` must have a unique monotonic increasing index and all p_totals > 0.

    Such a series comes from ``a.density_df.query('p_total > 0').p_total``, for example.

    Tested using numpy vs pd.Series lookup functions, and this version is much
    faster. See ``var_tvar_test_suite`` function below for testers (obviously
    run before this code was integrated).

    Changed in v. 0.13.0

    """

    # audits
    assert ser.index.is_unique, 'index values must be unique'
    assert ser.index.is_monotonic_increasing, 'index values must be increasing'

    # detach from the outside scope
    ser = ser.copy()

    # create needed arrays
    x_np = np.array(ser.index)
    # better not to cumulate array when all elements are equal (because of
    # floating point issues). This does make some difference. 
    if np.all(np.isclose(ser, ser.iloc[0], atol=2**-53)):
        d = 1 / len(ser)
        cser = pd.Series(np.linspace(d, 1, len(ser)), index=ser.index)
    else:
        cser = ser.cumsum()
    cser_F_np = cser.to_numpy()
    # detach the index values
    # cser_idx = pd.Index(cser.values)
    tvar_unconditional = ((ser * ser.index)[::-1].cumsum()[::-1]).to_numpy()

    # these last three are annoyting because np.where does not short circuit
    tvar_unconditional = np.hstack((tvar_unconditional, np.inf, np.inf))
    cser_F_np2 = np.hstack((cser_F_np, 1))
    x_np2l = np.hstack((x_np, x_np[-1]))
    x_np2u = np.hstack((x_np, np.inf))
    # x_max = cser_F_np[-2]

    # tests show this is about 6 times faster than
    # q = interp1d(cser, ser.index, kind='next', bounds_error=False, fill_value=(ser.index.min(), ser.index.max()))
    def q_lower(p):
        nonlocal x_np2l, cser_F_np
        return x_np2l[np.searchsorted(cser_F_np, p, side='left')]

    def q_upper(p):
        nonlocal x_np2u, cser_F_np
        return x_np2u[np.searchsorted(cser_F_np, p, side='right')]

    def tvar(p):
        """
        Vectorized TVaR computation.
        """
        nonlocal cser_F_np, x_np, tvar_unconditional
        if isinstance(p, (float, int)):
            # easy
            if p >= cser_F_np[-2]:
                return x_np[-1]
            else:
                idx = np.searchsorted(cser_F_np, p, side='right')
                return ((cser_F_np[idx] - p) * x_np[idx] + tvar_unconditional[idx + 1]) / (1 - p)
        else:
            # vectorized
            p = np.array(p)
            idx = np.searchsorted(cser_F_np, p, side='right')
            return np.where(idx >= len(cser_F_np) - 1,
                            x_np[-1],
                           ((cser_F_np2[idx] - p) * x_np2u[idx] + tvar_unconditional[idx + 1]) / (1 - p))

    QuantileFunctions = namedtuple("QuantileFUnctions", 'q q_lower var q_upper tvar')
    return QuantileFunctions(q_lower, q_lower, q_lower, q_upper, tvar)


def balanced_window(ser, p, bs=None):
    """Equal-tail window ``[q(p/2), q(1 - p/2)]`` of a realized pmf.

    The *post-calc* analogue of :func:`aggregate.distributions.estimate_agg_window`:
    where that places a window from method-of-moments fits *before* the FFT (a
    guess), this measures the window directly from an already-computed marginal
    pmf. A privileged consumer -- a bivariate aggregate, which runs its inner
    marginals first, or :meth:`Aggregate.focus` on a finished aggregate -- can
    therefore *measure* the support that matters rather than guess it.

    The window trims ``p / 2`` of the probability mass off **each** tail and
    keeps the central ``1 - p``. **Balanced** means equal *probability* each
    side, not equal value: a signed P&L marginal stays centred on its mass, and
    a skewed marginal trims more value off its heavy side but the same
    probability either way. The kept window always contains at least ``1 - p``
    of the mass (each discarded tail is at most ``p / 2``).

    Parameters
    ----------
    ser : pandas.Series
        A realized pmf: index = outcomes (``xs``), values = probabilities. The
        index must be unique and monotonic increasing and the values should sum
        to ~1 (the usual ``density_df.query('p_total > 0').p_total`` shape). The
        equal-tail accounting assumes the values are non-negative.
    p : float
        Total discarded tail mass, split equally between the two tails (``p / 2``
        each). Small, e.g. ``1e-6``; ``1 - p`` is the mass kept. Must satisfy
        ``0 < p < 1``. Unlike the coverage ``p`` of ``estimate_agg_window`` this
        is the *complementary* (discarded) mass, so there is no ``> 1 -> nines``
        reading here -- pass the literal tail mass.
    bs : float, optional
        Bucket size to snap the window to. The lower edge is floored and the
        upper edge ceiled to a multiple of ``bs`` so the window aligns with (and
        slightly contains) the grid. When ``None`` (default) the edges are
        returned exactly as the quantiles -- already grid points when ``ser``
        comes from a bucketed ``density_df``.

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper window edges. Both are **lower** quantiles
        (:func:`make_var_tvar` ``q_lower``): ``lo = q(p/2)``,
        ``hi = q(1 - p/2)``, matching :meth:`Aggregate.q` ``kind='lower'``.

    Notes
    -----
    Reuses :func:`make_var_tvar` for the quantiles rather than re-deriving a
    cumulative lookup, so the convention matches ``Aggregate.q`` exactly. With
    lower quantiles on both edges the mass strictly below ``lo`` is ``< p/2`` and
    the mass strictly above ``hi`` is ``<= p/2``, so the kept mass is
    ``>= 1 - p`` -- the guarantee the docstring promises.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f'p must be in (0, 1); got {p}.')
    qf = make_var_tvar(ser)
    lo = float(qf.q_lower(p / 2.0))
    hi = float(qf.q_lower(1.0 - p / 2.0))
    if bs:
        lo = np.floor(lo / bs) * bs
        hi = np.ceil(hi / bs) * bs
    return lo, hi


def kaplan_meier(df, loss='loss', closed='closed'):
    """
    Compute Kaplan Meier Product limit estimator based on a sample
    of losses in the dataframe df. For each loss you know the current
    evaluation in column ``loss`` and a 0/1 indicator for open/closed
    in ``closed``.

    The output dataframe has columns

    * index x_i, size of loss
    * open - the number of open events of size x_i (open claim with this size)
    * closed - the number closed at size x_i
    * events - total number of events of size x_i
    * n - number at risk at x_i
    * s - probability of suriviving past x_i = 1 - closed / n
    * pl - cumulative probability of surviving past x_i

    See ipython workbook kaplan_meier.ipynb for a check against lifelines
    and some kaggle data (telco customer churn,
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
    https://towardsdatascience.com/introduction-to-survival-analysis-the-kaplan-meier-estimator-94ec5812a97a

    :param df: dataframe of data
    :param loss: column containing loss amount data
    :param closed: column indicating if the obervation is a closed claim (1) or open (0)
    :return: dataframe as described above
    """

    df = df[[loss, closed]].rename(columns={loss: 'loss', closed: 'closed'}).copy()
    df['open'] = 1 - df.closed
    df = df.sort_values(['loss', 'closed'], ascending=[False, True]).reset_index(drop=True)

    df = df.groupby(['loss', 'closed']).count()
    # c has index loss amount and closed indicator, and column number of observations
    c = df.unstack(1)
    # total number of observables at each loss event size
    c['t'] = c.sum(1)
    # total number at risk at each event size
    c['n'] = c.t[::-1].cumsum()
    # better column names
    c.columns = ['open', 'closed', 'events', 'n']
    #
    c = c.fillna(0)
    # prob of surviving past each observed event size
    c['s'] = 1 -  c.closed / c.n
    # KM product estimator
    c['pl'] = c.s.cumprod()
    return c


def kaplan_meier_np(loss, closed):
    """
    Feeder to kaplan_meier where loss is np array  of loss amounts and
    closed a same sized array of 0=open, 1=closed indicators.
    """
    df = pd.DataFrame({'loss': loss, 'closed': closed})
    return kaplan_meier(df)


def agg_help(self, regex):
    """
    Investigate self for matches to the regex. If callable, try calling with no args, else display.

    Fka ``more``.

    Module-level free function backing the ``.help(regex)`` method on
    :class:`Aggregate`, :class:`Portfolio`, and :class:`Underwriter`. Named
    ``agg_help`` (not ``help``) to avoid shadowing Python's builtin ``help``
    at module / package scope.
    """
    # IPython imported lazily to keep it off the `import aggregate` path
    # (it is ~1s to import); see module note below.
    from IPython.display import Markdown, display
    for i in dir(self):
        if re.search(regex, i):
            ob = getattr(self, i)
            if not callable(ob):
                display(Markdown(f'### Attribute: {i}\n'))
                display(ob)
            else:
                display(Markdown(f'### Callable: {i}\n'))
                try:
                    print(ob())
                except Exception:
                    help(ob)


def introspect(ob):
    """
    Discover the non-private methods and properties of an object ``ob``.

    Used to build the class cheat sheets: it walks ``dir(ob)``, classifies each
    public name as a method, property, or plain field, and records its type,
    value, call signature, and docstring. The result is a :class:`pandas.DataFrame`
    sorted by classification then name, ready to export and arrange by category.

    Parameters
    ----------
    ob : object
        Any instance. Typically an :class:`Underwriter`, :class:`Aggregate`,
        :class:`Portfolio`, :class:`Severity`, or :class:`Distortion`.

    Returns
    -------
    pandas.DataFrame
        One row per public name with columns ``name``, ``kind`` (``method``,
        ``property``, ``field``, or ``error``), ``value``, ``type``,
        ``signature``, ``help`` (first docstring line), and ``length``.

    Notes
    -----
    Accessing a property can raise — e.g. a :class:`Portfolio` property that
    assumes the object has been updated. A bare ``getattr`` in the loop would
    then abort the whole introspection (the historical reason this function
    "would not run" on a Portfolio). Each access is therefore guarded: a name
    whose access raises is reported with ``kind='error'`` and the exception text
    in ``value``, so introspection always completes and surfaces the offender
    instead of dying on it. Classification (property vs. field) is read from the
    *class* via ``getattr(type(ob), name)`` so it does not depend on the instance
    access succeeding.
    """
    names = [i for i in dir(ob) if i[0] != '_']
    rows = []
    for name in names:
        class_attr = getattr(type(ob), name, None)
        is_property = isinstance(class_attr, property)
        # guarded access: a raising property must not abort the whole walk
        try:
            g = getattr(ob, name)
        except Exception as e:  # noqa: BLE001 - report, don't propagate
            rows.append([name, 'error', f'{type(e).__name__}: {e}', '', '', '', 0])
            continue

        value = ''
        type_str = ''
        signature = ''
        help_str = ''
        length = 0
        if callable(g) and not is_property:
            kind = 'method'
            try:
                signature = str(inspect.signature(g))
            except (TypeError, ValueError):
                signature = ''
            help_str = (g.__doc__ or '').strip().split('\n')[0]
        else:
            kind = 'property' if is_property else 'field'
            value = str(g)
            type_str = str(type(g))
            try:
                length = len(g)
            except TypeError:
                length = 0

        rows.append([name, kind, value, type_str, signature, help_str, length])

    df = pd.DataFrame(
        rows,
        columns=['name', 'kind', 'value', 'type', 'signature', 'help', 'length'])
    df = df.sort_values(['kind', 'length', 'name']).reset_index(drop=True)
    return df


def explain_validation(rv):
    """
    Explain the validation result rv.
    Don't over report: if you fail CV don't need to be told you fail Skew too.

    Under reinsurance the realised view has no independent theoretical and
    cannot be validated, but the SUBJECT (gross) view was validated under
    the hood (§1.3 of the aggregate refactor plan). The message reports
    that subject status alongside the ``reinsurance`` marker, so the user
    can tell whether the underlying gross object is sound.
    """
    if rv == Validation.NOT_UNREASONABLE:
        return "not unreasonable"
    if rv & Validation.NOT_UPDATED:
        return "n/a, not updated"
    # Collect failures from the SEV/AGG/ALIASING flags (suppressing higher
    # moments once a lower-order moment already failed).
    parts = []
    if rv & Validation.SEV_MEAN:
        parts.append('sev mean')
    if rv & Validation.AGG_MEAN:
        parts.append('agg mean')
    if rv & Validation.ALIASING:
        parts.append('agg mean error >> sev, possible aliasing; try larger bs')
    if not (rv & Validation.SEV_MEAN) and (rv & Validation.SEV_CV):
        parts.append('sev cv')
    if not (rv & Validation.AGG_MEAN) and (rv & Validation.AGG_CV):
        parts.append('agg cv')
    if not (rv & Validation.SEV_CV) and (rv & Validation.SEV_SKEW):
        parts.append('sev skew')
    if not (rv & Validation.AGG_CV) and (rv & Validation.AGG_SKEW):
        parts.append('agg skew')
    explanation = ', '.join(parts)
    if rv & Validation.REINSURANCE:
        if explanation:
            return f'reinsurance; subject fails {explanation}'
        return 'reinsurance; subject not unreasonable'
    return f'fails {explanation}'


