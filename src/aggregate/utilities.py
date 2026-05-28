from collections import namedtuple
import itertools
import logging
from numbers import Number

import numpy as np
import pandas as pd
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
import re
import scipy.fft as sft
from scipy.interpolate import interp1d
from IPython.display import HTML, Markdown, display

from .constants import Validation


logger = logging.getLogger(__name__)

__all__ = [
    'decl_pprint',
    'ft', 'ift',
    'subsets',
    'round_bucket',
    'make_ceder_netter', 'nice_multiple',
    'qd', 'mv',
    'make_var_tvar', 'kaplan_meier', 'kaplan_meier_np',
    'agg_help', 'explain_validation',
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


def decl_pprint(txt, split=0, html=False, show=True):
    """
    Try to format an agg program. This is difficult because of dfreq and dsev, optional
    reinsurance, etc. Go for a simple approach of removing unnecessary spacing
    and removing notes. Notes can be accessed from the spec that is always to hand.

    For long programs use split=60 or so, they are split at appropriate points.

    Best to use html = True to get colorization.

    :param txt: program text input
    :param split: if > 0 split lines at this length
    :param html: if True return html (via pygments) , else return text
    """
    ans = []
    # programs come in as multiline
    txt = txt.replace('\n\tagg', ' agg')
    for t in txt.split('\n'):
        clean = re.sub(r'[ \t]+', ' ', t.strip())
        clean = re.sub(r' note\{[^}]*\}', '', clean)
        if split > 0 and len(clean) > split:
            clean = re.sub(
                r' ((dfreq )([0-9]+ )|([0-9]+ )(claims?|premium|loss|exposure)'
                r'|d?sev|dfreq|occurrence|agg|aggregate|wts?|mixed|poisson|fixed)',
                           r'\n  \1', clean)
        if clean[:4] == 'port':
            # put in extra tabs at agg for portfolios
            sc = clean.split('\n')
            clean = sc[0] + '\n' + '\n'.join([i if i[:5] == '  agg' else '  ' + i for i in sc[1:]])
        ans.append(clean)
    ans = '\n'.join(ans)
    if html is True:
        # ans = f'<p><code>{ans}\n</code></p>'
        # notes = re.findall('note\{([^}]*)\}', txt)
        # for i, n in enumerate(notes):
        #     ans += f'<p><small>Note {i+1}. {n}</small><p>'
        # use pygments to colorize
        agg_lex = get_lexer_by_name('agg')
        # remove extra spaces
        txt = re.sub(r'[ \t\n]+', ' ', txt.strip())
        ans = HTML(highlight(txt, agg_lex, HtmlFormatter(style='friendly', full=False)))
    if show:
        print(ans)
        return
    else:
        return ans


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




def round_bucket(bs):
    """
    Compute a decent rounded bucket from an input float ``bs``. ::

        if bs > 1 round to 2, 5, 10, ...

        elif bs < 1 find the smallest power of two greater than bs

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

    if bs == 1:
        return bs

    if bs > 1:
        # rounded bs, to an integer
        rbs = np.round(bs, 0)
        if rbs == 1:
            return 2.0
        elif rbs == 2:
            return 2
        elif rbs <= 5:
            return 5.0
        elif rbs <= 10:
            return 10.0
        else:
            rbs = np.round(bs, -int(np.log10(bs)))
            if rbs < bs:
                rbs *= 2
            return rbs

    if bs < 1:
        # inverse bs
        # originally
        # bsi = 1 / bs
        # nbs = 1
        # while nbs < bsi:
        #     nbs <<= 1
        # nbs >>= 1
        # return 1. / nbs
        # same answer but ? clearer and slightly quicker
        x = 1. / bs
        x = bin(int(x))
        x = '0b1' + "0" * (len(x) -3)
        x = int(x[2:], 2)
        return 1./ x


def make_ceder_netter(reins_list, debug=False):
    """
    Build the netter and ceder functions. It is applied to occ_reins and agg_reins,
    so should be stand-alone.

    The reinsurance functions are piecewise linear functions from 0 to inf with
    kinks as needed to express the ceded loss as a function of subject (gross) loss.

    The entries in ``reins_list`` are tuples (share of, limit, attach) where share of is the
    percentage share, between 0 and 1.

    For example, if ``reins_list = [(1, 10, 0), (0.5, 30, 20)]`` the program is 10 x 10 and
    15 part of 30 x 20 (share=0.5). This requires nodes at 0, 10, 20, 50, and inf.

    It is easiest to make the ceder function. Ceded loss at subject loss at x equals
    the sum of the limits below x plus the cession to the layer in which x lies. The
    variable ``base`` keeps track of the layer, ``h`` of the sum (height) of lower layers.
    ``xs`` tracks the knot points, ``ys`` the values.

    ::

         Break (xs)   Ceded (ys)
              0            0
             10            0
             20           10
             50           25
            inf           25


    For example:
    ::

        %%sf 1 2

        c, n, x, y = make_ceder_netter([(1, 10, 10), (0.5, 30, 20), (.25, np.inf, 50)], debug=True)

        xs = np.linspace(0,250, 251)
        ys = c(xs)

        ax0.plot(xs, ys)
        ax0.plot(xs, xs, ':C7')
        ax0.set(title='ceded')

        ax1.plot(xs, xs-ys)
        ax1.plot(xs, xs, 'C7:')
        ax1.set(title='net')

    :param reins_list: a list of (share of, limit, attach), e.g., (0.5, 3, 2) means 50% share of 3x2
        or, equivalently, 1.5 part of 3 x 2. It is better to store share rather than part
        because it still works if limit == inf.
    :param debug: if True, return layer function xs and ys in addition to the interpolation functions.
    :return: netter and ceder functions; optionally debug information.
    """
    # poor mans inf
    INF = 1e99
    h = 0
    base = 0
    xs = [0]
    ys = [0]
    for (share, y, a) in reins_list:
        # part of = share of times limit
        if np.isinf(y):
            y = INF
        p = share * y
        if a > base:
            # moved to new layer, write out left-hand knot point
            xs.append(a)
            ys.append(h)
        # increment height
        h += p
        # write out right-hand knot points
        xs.append(a + y)
        ys.append(h)
        # update left-hand end
        base += (a + y)
    # if not at infinity, stay flat from base to end
    if base < INF:
        xs.append(np.inf)
        ys.append(h)
    ceder = interp1d(xs, ys)
    netter = lambda x: x - ceder(x)
    if debug:
        return ceder, netter, xs, ys
    else:
        return ceder, netter


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
            if 'Err CV(X)' in x.describe.columns:
                qd(x.describe.drop(columns=['Err CV(X)']).fillna(''), accuracy=accuracy, **kwargs)
            else:
                # object not updated
                qd(x.describe.fillna(''), accuracy=accuracy, **kwargs)
            bss = 'na' if x.bs == 0 else (f'{x.bs:.0f}' if x.bs >= 1 else f'1/{1/x.bs:.0f}')
            vr = x.explain_validation()
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
                except Exception as e:
                    help(ob)


def explain_validation(rv):
    """
    Explain the validation result rv.
    Don't over report: if you fail CV don't need to be told you fail Skew too.
    """
    if rv == Validation.NOT_UNREASONABLE:
        return "not unreasonable"
    elif rv & Validation.NOT_UPDATED:
        return "n/a, not updated"
    elif rv & Validation.REINSURANCE:
        return "n/a, reinsurance"
    else:
        explanation = 'fails '
        if rv & Validation.SEV_MEAN:
            # explanation += f'sev mean: {ob.sev_m: .4e} vs {ob.est_sev_m: .4e}\n'
            explanation += f'sev mean, '
        if rv & Validation.AGG_MEAN:
            explanation += f'agg mean, '
        if rv & Validation.ALIASING:
            explanation += "agg mean error >> sev, possible aliasing; try larger bs, "
        if not(rv & Validation.SEV_MEAN) and (rv & Validation.SEV_CV):
            # explanation += f'sev cv: {ob.sev_cv: .4e} vs {ob.est_sev_cv: .4e}, '
            explanation += f'sev cv, '
        if not(rv & Validation.AGG_MEAN) and (rv & Validation.AGG_CV):
            explanation += f'agg cv, '
        if not (rv & Validation.SEV_CV) and (rv & Validation.SEV_SKEW):
            # explanation += f'sev skew: {ob.sev_skew: .4e} vs {ob.est_sev_skew: .4e}, '
            explanation += f'sev skew, '
        if not (rv & Validation.AGG_CV) and (rv & Validation.AGG_SKEW):
            explanation += f'agg skew, '
    return explanation[:-2]


