# hacked out of case_studies


class Ratings():
    """
    class to hold various ratings dictionaries
    Just facts

    """
    # https://www.spglobal.com/ratings/en/research/articles/200429-default-transition-and-recovery-2019-annual-global-corporate-default-and-rating-transition-study-11444862
    # Table 9 On Year Global Corporate Default Rates by Rating Modifier
    # in PERCENT
    sp_ratings = 'AAA    AA+  AA    AA-   A+    A     A-    BBB+  BBB   BBB-  BB+   BB    BB-   B+    B     B-    CCC/C'
    sp_default = '0.00  0.00  0.01  0.02  0.04  0.05  0.07  0.12  0.21  0.25  0.49  0.70  1.19  2.08  5.85  8.77  24.34'

    @classmethod
    def make_ratings(cls):
        sp_ratings = re.split(' +', cls.sp_ratings)
        sp_default = [np.round(float(i) / 100, 8) for i in re.split(' +', cls.sp_default)]
        spdf = pd.DataFrame({'rating': sp_ratings, 'default': sp_default})
        return spdf


def make_cap_table_old(self, kind='gross'):
    """
    Suggest reasonable debt tranching for kind=(gross|net) subject to self.d2tc debt to total capital limit.
    Uses S&P bond default analysis.

    Creates self.cap_table and self.debt_stats

    This is a cluster. There must be a better way...

    This version did all the difficult tranching...which is based on dubious info anyways...

    """

    port = self.ports[kind]

    r = Ratings()
    spdf = r.make_ratings()
    # ad hoc adjustment for highest rated issues
    spdf.loc[0, 'default'] = .00003
    spdf.loc[1, 'default'] = .00005
    spdf['attachment'] = [port.q(1 - i) for i in spdf.default]
    spdf = spdf.set_index('rating', drop=True)
    # if you want a plot
    # spdf[::-1].plot.barh(ax=ax, width=.8)
    # ax.set(xscale='log', xlabel='default probability')
    # ax.legend().set(visible=False)

    # debt to total capital limit
    a, premium, el = self.pricing_summary.loc[['a', 'P', 'L'], kind]
    capital = a - premium
    debt = self.d2tc * capital
    equity = capital - debt
    debt_attach = a - debt
    prob_debt_attach = port.sf(debt_attach)
    i = (spdf.default < prob_debt_attach).values.argmin()
    j = (spdf.default > 1 - self.reg_p).values.argmax() - 1
    attach_rating = spdf.index[i]
    exhaust_rating = spdf.index[j]
    # tranche out: pull out the relevant rating bands
    bit = spdf.iloc[j:i + 1]
    # add debt attachment and capital
    bit.loc['attach'] = [prob_debt_attach, debt_attach]
    bit.loc['capital'] = [port.sf(a), a]
    bit = bit.sort_values('attachment', ascending=False)

    # compute tranche widths and extract just the tranches that apply (tricky!)
    # convert to series with attachment
    tranches = bit.attachment.shift(1) - bit.attachment
    # tranches ott ratings, capital, ratings, attach, bottom rating
    # needs converting into just the relevant ratings bands. Attach is replaced with the bottom rating
    # and cut off below capital
    # hence
    ix = list(tranches.index[:-1])
    ix[-1] = tranches.index[-1]
    # drop bottom
    tranches = tranches.iloc[:-1]
    # re index
    tranches.index = ix
    capix = tranches.index.get_loc('capital')
    tranches = tranches.iloc[capix + 1:]
    tranches.index.name = 'rating'
    tranches = tranches.to_frame()
    tranches.columns = ['Amount']
    # merge in attachments
    tranches['attachment'] = bit.iloc[capix + 1:-1, -1].values
    # tranches

    # integrate into a cap table
    cap_table = tranches.copy()
    # add prob attaches
    cap_table['Pr Attach'] = [port.sf(i) for i in cap_table.attachment]
    # add equity, premium (margin), and EL rows
    cap_table.loc['Equity'] = [debt_attach - premium, premium, port.sf(premium)]
    cap_table.loc['Margin'] = [premium - el, el, port.sf(el)]
    cap_table.loc['EL'] = [el, 0, port.sf(0)]

    cap_table['Total'] = cap_table.Amount[::-1].cumsum()
    cap_table['Pct Assets'] = cap_table.Amount / cap_table.iloc[0, -1]
    cap_table['Cumul Pct'] = cap_table.Total / cap_table.iloc[0, -2]
    # just recompute to be sure...
    cap_table['Pr Exhaust'] = [port.sf(i) for i in cap_table.Total]
    cap_table.columns = ['Amount', 'Attaches', 'Pr Attaches', 'Exhausts', 'Pct Assets', 'Cumul Pct', 'Pr Exhausts']
    cap_table = cap_table[
        ['Amount', 'Pct Assets', 'Attaches', 'Pr Attaches', 'Exhausts', 'Pr Exhausts', 'Cumul Pct']]

    cap_table.columns.name = 'Quantity'
    cap_table.index.name = 'Tranche'

    # make the total and incremental views
    if port.augmented_df is not None:
        # first call to create cap table is before any pricing....
        total_renamer = {'F': 'Adequacy',
                         'capital': 'Capital',
                         'exa_total': 'Loss',
                         'exag_total': 'Premium',
                         'margin': 'Margin',
                         'lr': 'LR',
                         'coc': 'CoC',
                         'loss': 'Assets'}
        bit = port.augmented_df.loc[[port.snap(i) for i in cap_table.Exhausts],
                                 ['loss', 'F', 'exa_total', 'exag_total']].sort_index(ascending=False)
        bit['lr'] = bit.exa_total / bit.exag_total
        bit['margin'] = (bit.exag_total - bit.exa_total)
        bit['capital'] = bit.loss - bit.exag_total
        bit['coc'] = bit.margin / bit.capital
        bit['Discount'] = bit.coc / (1 + bit.coc)
        # leverage here does not make sense because of reserves
        bit.index = cap_table.index
        bit = bit.rename(columns=total_renamer)
        self.cap_table_total = bit

        marginal_renamer = {
                            'F': 'Adequacy',  # that the loss is in the layer
                            'loss': 'Assets',
                            'exa_total': 'Loss',
                            'exag_total': 'Premium',
                            'margin': 'Margin',
                            'capital': 'Capital',
                            'lr': 'LR',
                            'coc': 'CoC',
                            }
        bit = port.augmented_df.loc[[0] + [port.snap(i) for i in cap_table.Exhausts],
                                 ['loss', 'F', 'exa_total', 'exag_total']].sort_index(ascending=False)
        bit = bit.diff(-1).iloc[:-1]
        bit['lr'] = bit.exa_total / bit.exag_total
        bit['margin'] = (bit.exag_total - bit.exa_total)
        bit['capital'] = bit.loss - bit.exag_total
        bit['coc'] = bit.margin / bit.capital

        bit.index = self.cap_table.index
        bit.loc['Total', :] = bit.sum()
        bit.loc['Total', 'lr'] = bit.loc['Total', 'exa_total'] / bit.loc['Total', 'exag_total']
        bit.loc['Total', 'coc'] = bit.loc['Total', 'margin'] / bit.loc['Total', 'capital']
        bit['Discount'] = bit.coc / (1 + bit.coc)
        bit = bit.rename(columns=marginal_renamer)
        self.cap_table_marginal = bit

    # return ans, tranches, bit, spdf, cap_table
    self.debt_stats = pd.Series(
        [a, el, premium, capital, equity, debt, debt_attach, prob_debt_attach, attach_rating, exhaust_rating],
        index=['a', 'EL', 'P', 'Capital', 'Equity', 'Debt', 'D_attach', 'Pr(D_attach)', 'Attach Rating',
               'Exhaust Rating'])
    self.cap_table = cap_table
    self.sp_ratings = spdf



@staticmethod
def default_float_format(x, neng=3):
    """
    the endless quest for the perfect float formatter...

    Based on Great Tikz Format

    tester::

        for x in 1.123123982398324723947 * 10.**np.arange(-23, 23):
            print(default_float_format(x))

    :param x:
    :return:
    """
    ef = EngFormatter(neng, True)
    try:
        if np.isnan(x):
            return ''
        elif x == 0:
            ans = '0'
        elif 1e-3 <= abs(x) < 1e6:
            if abs(x) < 1:
                ans = f'{x:.3g}'
            elif abs(x) < 10:
                ans = f'{x:.4g}'
            elif abs(x) <= 100:
                ans = f'{x:.4g}'
            elif abs(x) < 1000:
                ans = f'{x:,.1f}'
            else:
                ans = f'{x:,.0f}'
        else:
            ans = ef(x)
        return ans
    except ValueError:
        return x

@staticmethod
def default_float_format2(x, neng=3):
    """
    the endless quest for the perfect float formatter...
    Like above, but two digit numbers still have 3dps
    Based on Great Tikz Format

    tester::

        for x in 1.123123982398324723947 * 10.**np.arange(-23, 23):
            print(default_float_format(x))

    :param x:
    :return:
    """
    ef = EngFormatter(neng, True)
    try:
        if np.isnan(x):
            return ''
        elif abs(x) <= 1e-14:
            ans = '0'
        elif 1e-3 <= abs(x) < 1e6:
            if abs(x) < 1:
                ans = f'{x:.3g}'
            elif abs(x) < 10:
                ans = f'{x:.4g}'
            elif abs(x) < 100:
                ans = f'{x:.5g}'
            elif x == 100:
                ans = '100'
            elif abs(x) < 1000:
                ans = f'{x:,.1f}'
            else:
                ans = f'{x:,.0f}'
        else:
            ans = ef(x)
        return ans
    except ValueError:
        return x

def bond_pricing_table(self):
    """
    Credit yield curve info used to create calibrated blend

    creates dd, the distortion dataframe

    @return:
    """
    dd = self.sp_ratings.copy()
    dd['yield'] = np.nan
    # TODO SUBOPTIMAL!! Interpolate yields
    dd.loc[['AAA', 'AA', 'A', 'A-', 'BBB+', 'BBB', 'B+', 'CCC/C'], 'yield'] = [0.0364, .04409, .04552, .04792,
                                                                               .04879, .05177, .09083, .1]
    dd['yield'] = dd.set_index('default')['yield'].interpolate(method='index').values

    lowest_tranche = self.cap_table.index[-4]
    # +2 to go one below the actual lowest tranche used in the debt structure
    lowest_tranche_ix = self.sp_ratings.index.get_loc(lowest_tranche) + 2

    dd = dd.iloc[0:lowest_tranche_ix]
    dd = dd.drop(columns=['attachment'])
    return dd



def make_blend(self, kind='gross', debug=False):
    """
    blend_d0 is the Book's blend, with roe above the equity point
    blend_d is calibrated to the same premium as the other distortions

    method = extend if f_blend_extend or ccoc
        ccoc = pick and equity point and back into its required roe. Results in a
        poor fit to the calibration data

        extend = extrapolate out the last slope from calibrtion data

    Initially tried interpolating the bond yield curve up, but that doesn't work.
    (The slope is too flat and it interpolates too far. Does not look like
    a blend distortion.)
    Now, adding the next point off the credit yield curve as the "equity"
    point and solving for ROE.

    If debug, returns more output, for diagnostics.

    """
    global logger

    # otherwise, calibrating to self.ports[kind]
    port = self.ports[kind]
    # dd = distortion dataframe; this is already trimmed down to the relevant issues
    dd = self.bond_pricing_table()
    attach_probs = dd.iloc[:, 0]
    layer_roes = dd.iloc[:, 1]

    # calibration prefob
    df = port.density_df
    a = self.pricing_summary.at['a', kind]
    premium = self.pricing_summary.at['P', kind]
    logger.info(f'Calibrating to premium of {premium:.1f} at assets {a:.1f}.')
    # calibrate and apply use 1 = forward sum
    bs = self.gross.bs
    S = (1 - df.p_total[0:a-bs].cumsum())

    d = None

    # generic NR function
    def f(s):
        nonlocal d
        eps = 1e-8
        d = make_distortion(s)
        d1 = make_distortion(s + eps / 2)
        d2 = make_distortion(s - eps / 2)
        ex = pricer(d)
        p1 = pricer(d1)
        p2 = pricer(d2)
        ex_prime = (p1 - p2) / eps
        return ex - premium, ex_prime

    def pricer(distortion):
        # re-state as series, interp returns numpy array
        return np.sum(distortion.g(S)) * bs
        # temp = pd.Series(distortion.g(S)[::-1], index=S.index)
        # temp = temp.shift(1, fill_value=0).cumsum() * bs
        # return temp.iloc[-1]

    # two methods of calibration
    if self.f_blend_extend is False:

        def make_distortion(roe):
            nonlocal attach_probs, layer_roes
            layer_roes[-1] = roe
            g_values = (attach_probs + layer_roes) / (1 + layer_roes)
            return agg.Distortion.s_gs_distortion(attach_probs, g_values, 'blend')

        # newton raphson with numerical derivative
        i = 0
        # first step
        s = self.roe
        fx, fxp = f(s)
        logger.info(f'starting premium {fx + premium:.1f}\ttarget={premium:.1f} @ {s:.3%}')
        max_iter = 50
        logger.info('  i       fx        \troe        \tfxp')
        logger.info(f'{i: 3d}\t{fx: 8.3f}\t{s:8.6f}\t{fxp:8.3f}')

    elif self.f_blend_extend is True:
        pp = interp1d(dd.default, dd['yield'], bounds_error=False, fill_value='extrapolate')

        def make_distortion(s):
            nonlocal attach_probs, layer_roes
            attach_probs[-1] = s
            layer_roes[-1] = pp(s)
            g_values = (attach_probs + layer_roes) / (1 + layer_roes)
            return agg.Distortion.s_gs_distortion(attach_probs, g_values, 'blend')

        # newton raphson with numerical derivative
        i = 0
        # first step, start a bit to the right of the largest default used in pricing
        s = dd.default.max() * 1.5
        fx, fxp = f(s)
        logger.info(f'starting premium {fx + premium:.1f}\ttarget={premium:.1f} @ {s:.3%}')
        max_iter = 50
        logger.info('  i       fx        \ts          \tfxp')
        logger.info(f'{i: 3d}\t{fx: 8.3f}\t{s:8.6f}\t{fxp:8.3f}')

    else:
        raise ValueError(f'Inadmissible option passed to make_blend.')

    # actual NR code is generic
    while abs(fx) > 1e-8 and i < max_iter:
        logger.info(f'{i: 3d}\t{fx: 8.3f}\t{s:8.6f}\t{fxp:8.3f}')
        s = s - fx / fxp
        fx, fxp = f(s)
        i += 1
    if i == max_iter:
        logger.error(f'NR failed to converge...Target={premium:2f}, achieved={fx + premium:.2f}')
    logger.info(f'Ending parameter={s:.5g} (s or roe)')
    logger.info(f'Target={premium:2f}, achieved={fx + premium:.2f}')

    if debug is True:
        return d, premium, pricer
    else:
        return d