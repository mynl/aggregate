"""

Cut out hacks aimed at reproducing book exhibits (color, roe hack, blend fix etc.) these are
now all applied by default. Set up for new syntax case specification.

* No case manager - that is site only. No netter.
* gridoff removed and grid lines cut out.
* locf not used
* make_netter and book style calls

Trimeed out Tense and Relaxed portfolios.

COPIED FROM WEBSITE/PricingInsuranceRisk
JUNE 26 2022
For hacking and planned integration into aggregate

NO BACKWARD COMPATIBILITY REQUIREMENT... though ability to generate book examples a plus

(PIR version came from spectral risk mono/Python ... which came from the ipynb exhibit creator)

Note and To Do

1. Per occ for HuSCS? This is actually quite hard...
2. Other
"""

# Book Case Studies in library
# See case_studies_runner.py to run
# Integrates code from common_header, common_scripts, and hack
# Integrates FigureManager, to be stand-alone

import aggregate as agg
from aggregate.distr import Aggregate
from aggregate.port import Portfolio
from aggregate.utils import round_bucket, make_mosaic_figure, GreatFormatter
import argparse
from collections import OrderedDict
from cycler import cycler
from datetime import datetime
import hashlib
from itertools import product
from jinja2 import Environment, FileSystemLoader
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from numbers import Number
import numpy as np
from numpy.linalg import pinv
import os
import pandas as pd
from pandas.io.formats.format import EngFormatter
from pathlib import Path
from platform import platform
import psutil
import re
import scipy.stats as ss
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import shlex
from subprocess import Popen
from textwrap import fill
from titlecase import titlecase as title
import types
import warnings

from IPython.display import HTML, display

# general set up
pd.set_option("display.float_format", EngFormatter(3, True))
pd.set_option('display.max_rows', 500)

# get the logger
logger = logging.getLogger('aggregate')

# logging.captureWarnings(True)  -> all warnings emitted by the warnings module
# will automatically be logged at level WARNING
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# utilities
stat_renamer = {'L': 'Expected Loss', 'P': "Premium", 'LR': 'Loss Ratio',
                'M': "Margin", 'PQ': 'Leverage', 'Q': 'Capital', 'ROE': "ROE", 'a': 'Assets',
                'rp': "Return Period", 'epdr': 'EPD Ratio', 'EPDR': 'EPD Ratio'}


def distortion_namer(x):
    if len(x) < 4:
        return x
    x = x.split(',')[0]
    x = x[3:]
    x = {'Wang-normal': "Wang",
         "Proportional Hazard": "PH",
         "Constant ROE": "CCoC",
         "roe": 'CCoC',
         "blend": "Blend",
         "Tail VaR": "TVaR",
         "Dual Moment": "Dual"}[x]
    return x


def universal_renamer(x):
    """

    """
    abbreviations = dict(LR='Loss Ratio', L='Loss', P='Premium', M='Margin', Q='Capital', a='Assets', PQ='Leverage',
                         ROE='Rate of Return',
                         MM='Miller-Modigliani', ER='Expense Ratio', EL='Expected Loss', CR='Combined Ratio',
                         RP='Return Period',
                         EmpMean='Net', HS='Hu-SCS')
    # expand pure abbreviations, only do one replacement
    if isinstance(x, Number):
        return x

    x = abbreviations.get(x, x)
    # de underscore
    x = x.replace('p_', '')
    x = x.replace('_', ' ')
    # stip off starting T. and V.
    x = re.sub(r'^([TMV])\.', lambda m: {
        'M': 'Marginal ', 'T': 'Primary ', 'V': 'Excess '}.get(m[1]), x)
    # sensible title case
    x = title(x)
    # deal with embedded abbreviations (e.g. plan lr -> Plan Lr or Plan ROE -> Plan Roe)
    x = re.sub(r'\b(Roe|Cv|Lr|Rol|Uw|Epd|Ph|Hs)\b', lambda x: x[1].upper(), x)
    x = x.replace('Scaled', 'Scaled ')
    x = x.replace('Sop', 'SoP')
    x = x.replace('EqRisk', 'Equal Risk ')
    x = x.replace('MerPer', 'Merton-Perold ')
    return x


def urn(df):
    """ apply universal renamer """
    return df.rename(index=universal_renamer).rename(columns=universal_renamer)


class CaseStudy(object):
    _stats_ = ['L', 'M', 'P', 'LR', 'Q', 'ROE', 'PQ', 'a']
    _dist_ = ['EL', 'Dist roe', 'Dist ph', 'Dist wang', 'Dist dual', 'Dist tvar', 'Dist blend', ]
    _classical_ = ['EL', 'ScaledEPD', 'ScaledTVaR', 'ScaledVaR', 'EqRiskEPD', 'EqRiskTVaR', 'EqRiskVaR',
                   'coTVaR', 'covar']

    def __init__(self):
        """
        Create an empty CaseStudy.

        Use various factory options to actually populate.

        """

        # variables set in other functions
        self.blend_d = None
        self.roe_d = None
        self.dist_dict = None
        self.uw = agg.Underwriter(create_all=False, update=False)

        self.tab13_1 = None
        self.sop = None
        self.lrs = None
        self.classic_pricing = None
        self.lns = None
        self.prices = None
        self.dgross = None
        self.dnet = None
        self.distortion_pricing = None
        self.classical_pricing = None
        self.summaries = None
        self.sp_ratings = None
        self.cap_table = None
        self.debt_stats = None
        self.walk = None
        self.progression = None
        self.p_stars = None
        self.diff_g = None
        self.exeqa = None
        self.boundss = None
        self.ad_compss = None

        self.case_id = ""
        self.case_name = ""
        self.case_description = ""
        self.a_name = ""
        self.b_name = ""
        self.a_distribution = ""
        self.b_distribution = ""
        self.b_distribution_gross = ""
        self.b_distribution_net = ""
        self.re_line = ""
        self.re_type = ""
        self.re_description = ''
        self.reg_p = 0.
        self.roe = 0.
        self.d2tc = 0.
        self.f_discrete = False
        self.f_blend_extend = False
        self.bs = 0.
        self.log2 = 0
        self.padding = 1

        # graphics defaults
        self.fw = 3.5 * 1.333
        self.fh = 3.5
        color_mode = 'color'
        cycle_mode = 'c'
        self.smfig = FigureManager(cycle=cycle_mode, color_mode=color_mode, font_size=10, legend_font='small',
                                   default_figsize=(self.fw, self.fh))
        self.colormap = 'viridis'
        self.figure_bg_color = 'aliceblue'
        plt.rcParams["axes.facecolor"] = 'lightsteelblue'
        plt.rc('legend', fc='lightsteelblue', ec='lightsteelblue')

        # discounting and return
        self.v = 0.
        self.d = 0.
        self.gross = None
        self.net = None
        self.ports = None
        self.pricings = None
        self.re_summary = None

        # graphics and presentation
        self.show = True
        self.figtype = 'png'

        # output related
        self.cache_base = Path.home() / 'aggregate/cases'
        self.cache_base.mkdir(exist_ok=True)
        self.cache_dir = None

    def factory(self, case_id, case_name, case_description,
                     a_distribution, b_distribution_gross, b_distribution_net,
                     reg_p, roe, d2tc,
                     f_discrete, f_blend_extend, bs, log2, padding):
        """
        Create CaseStudy from case_id and all arguments in generic format with explicit reinsurance.

        f_blend_fix True by default and try to do extend method.
        f_roe_fix True by default (use approx to CCoC distortion)

        re_line is always line B.

        option_id = tame, cnc, hs, discrete, or a custom name that must be allowable as a directory name.
        portfolio_function(option_id, flags): function to create the portfolios and return all relevant variables

        d2tc = debt to total capital limit, for better blend distortion

        :param case_id:
        :param case_name:
        :param case_description:
        :param a_distribution:
        :param b_distribution_net:
        :param b_distribution_gross:
        :param reg_p:
        :param roe:
        :param d2tc:
        :param f_blend_extend: if true, use the extend method to parameterize blend; else ROE point method
        :param f_discrete:
        :param bs:
        :param log2:
        :param padding:
        """

        self.case_id = case_id  # originally option_id
        self.case_name = case_name
        self.case_description = case_description
        # programs
        self.a_distribution = a_distribution
        self.b_distribution_gross = b_distribution_gross
        self.b_distribution_net = b_distribution_net
        # derivable
        self.a_name = a_distribution.split(' ')[1]
        self.b_name = b_distribution_gross.split(' ')[1]
        self.re_line = self.b_name

        self.reg_p = reg_p
        self.roe = roe
        self.v = 1 / (1 + self.roe)
        self.d = 1 - self.v
        self.d2tc = d2tc
        self.f_discrete = f_discrete
        self.f_blend_extend = f_blend_extend
        self.log2 = log2
        self.bs = bs
        self.padding = padding

        # new style output from uw key (kind, name) output (obj or spec, program)
        out = self.uw(f'''
port Gross_{self.case_id}
    {self.a_distribution}
    {self.b_distribution_gross}
''')
        self.gross = out[('port', f'Gross_{self.case_id}')][0]
        # sort out better bucket
        if self.bs == 0 or np.isnan(self.bs):
            self.bs = self.gross.best_bucket(self.log2)
        self.gross.update(log2=self.log2, bs=self.bs, padding=self.padding, remove_fuzz=True)
        enhance_portfolio(self.gross)
        # self.gross = TensePortfolio(self.gross, ,
        #                             ROE=self.roe, p=self.reg_p)

        # net portfolio
        out = self.uw(f'''
port Net_{self.case_id}
    {self.a_distribution}
    {self.b_distribution_net}
''')
        self.net = out[('port', f'Net_{self.case_id}')][0]
        self.net.update(log2=self.log2, bs=self.bs, remove_fuzz=True)
        enhance_portfolio(self.net)
        self.ports = OrderedDict(gross=self.gross, net=self.net)

        self.pricings = OrderedDict()
        self.pricings['gross'] = pricing(self.gross, reg_p, roe)
        self.pricings['net'] = pricing(self.net, reg_p, roe)

        # are these used?
        self.re_type = self.net[self.re_line].reinsurance_kinds()
        self.re_description = self.net[self.re_line].reinsurance_description(kind='both')
        # TODO replace! Output as Table G in Ch 7
        # self.re_summary = pd.DataFrame([self.re_line, self.re_type, self.re_attach_p, self.re_attach, self.re_detach_p,
        #                                 self.re_detach - self.re_attach], index=[
        #     'Reinsured Line', 'Reinsurance Type', 'Attachment Probability', 'Attachment', 'Exhaustion Probability',
        #     'Limit'], columns=[self.case_id])
        # self.re_summary.index.name = 'item'

        # cap table needs pricing_summary
        self.make_audit_exhibits()

        # set up the common distortions
        # make the cap table
        if self.f_discrete is False:
            self.make_cap_table()
            try:
                self.blend_d = self.make_blend()
            except ValueError as e:
                logger.error(f'Extend method failed...defaulting back to book blend. Message {e}')
                # TODO try the ROE blend method?
                self.f_blend_extend = False
                self.blend_d = self.make_blend()

            logger.info('applying ROE approximation fix.')
            self.roe_d = self.approx_roe(e=1e-10)
            self.dist_dict = OrderedDict(roe=self.roe_d, blend=self.blend_d)
        else:
            # original book blend, FYI (if f_blend_fix = 0)
            attach_probs = np.array([1e-6, .01, .03, .05, .1, .2])
            layer_roes = np.array([.02, .03, .04, .06, .08, self.roe])
            g_values = (attach_probs + layer_roes) / (1 + layer_roes)
            self.blend_d = agg.Distortion.s_gs_distortion(attach_probs, g_values, 'blend')
            self.roe_d = self.approx_roe(e=1e-10)
            self.dist_dict = OrderedDict(roe=self.roe_d, blend=self.blend_d)

        self.cache_dir = self.cache_base / f'{self.case_id}'
        self.cache_dir.mkdir(exist_ok=True)

    def __repr__(self):
        return f'''Case Study object {self.case_id} @ {self.cache_dir} ({super().__repr__()})
Portfolios: {self.gross.name} (EL={self.gross.ex:.2f}) and {self.net.name} ({self.net.ex:.2f}).
Lines: {", ".join(self.gross.line_names)} (ELs={", ".join([f"{a.ex:.2f}" for a in self.gross])}).

'''

    def full_monty(self):
        """
        All updating and exhibit generation. No output. For use with command line.

        :param self:
        @return:
        """

        logger.log(35, 'Start Full Monty Update')
        self.make_all()
        process_memory()

        logger.log(35, 'display exhibits')
        self.show = False
        self.show_exhibits('all')
        process_memory()
        self.show_extended_exhibits()
        process_memory()

        logger.log(35, 'create graphics')
        self.show_graphs('all')
        process_memory()
        self.show_extended_graphs()
        process_memory()
        logger.log(35, f'{self.case_id} computed')

        # save the results
        mrm = ManualRenderResults(self)
        p, p1 = mrm.render()
        logger.log(35, f'{self.case_id} saved to HTML...complete!')

    def approx_roe(self, e=1e-15):
        """
        Make an approx to roe distortion with no mass

        :param e:
        @return:
        """
        aroe = pd.DataFrame({'col_x': [0, e, 1], 'col_y': [0, self.v * e + self.d, 1]})
        approx_roe_di = agg.Distortion('convex', None, None, df=aroe, col_x='col_x', col_y='col_y', display_name='roe')
        return approx_roe_di

    def make_audit_exhibits(self):
        # Chapter 7 Case Study Intro Exhibits

        # 1. MVSK
        # 2. VaR, TVaR, EPD, sop vs. total
        # 3. Density plots, linear and log scale
        # 4. Bivariate density plot
        # 5. TVaR and VaR Graphs
        cols = ['EmpMean', 'EmpCV', 'EmpSkew', 'EmpKurt',
                'P90.0', 'P95.0', 'P99.0', 'P99.6', 'P99.9']
        audit_all = pd.concat((port.audit_df.T.loc[cols] for port in self.ports.values(
        )), keys=self.ports.keys(), names=['view', 'line'], axis=1)
        audit_all.index.name = 'statistic'

        for k, port in self.ports.items():
            p = 0.975
            vd = port.var_dict(p, 'lower')
            s = pd.Series(list(vd.values()), index=[(k, j) for j in vd.keys()])
            audit_all.loc[f'VaR {100 * p:.1f}', s.index] = s
            vd = port.var_dict(p, 'tvar')
            s = pd.Series(list(vd.values()), index=[(k, j) for j in vd.keys()])
            audit_all.loc[f'TVaR {100 * p:.1f}', s.index] = s

            for p in [.9, .95, 0.975, .99, .996, .999]:
                vd = port.var_dict(p, 'tvar')
                s = pd.Series(list(vd.values()), index=[(k, j) for j in vd.keys()])
                audit_all.loc[f'TVaR {100 * p:.1f}', s.index] = s
                vd = port.var_dict(1 - p, 'epd')
                s = pd.Series(list(vd.values()), index=[(k, j) for j in vd.keys()])
                audit_all.loc[f'EPD {100 * (1 - p):.1f}', s.index] = s

        audit_all = audit_all.rename(
            index={f'P{i}': f'VaR {i:.1f}' for i in [
                90., 95., 0.975, 99., 99.6, 99.9]}
        ).rename(
            index=dict(EmpMean='Mean', EmpCV='CV',
                       EmpSkew='Skewness', EmpKurt='Kurtosis'))

        bit = audit_all.filter(regex='[RE]', axis=0).copy()
        if len(self.gross.line_names) == 2:
            ln1, ln2 = self.gross.line_names
            bit[('gross', 'sum')] = bit[('gross', ln1)] + bit[('gross', ln2)]
            bit[('net', 'sum')] = bit[('net', ln1)] + bit[('net', ln2)]
            bit[('gross', 'benefit')] = bit[('gross', 'sum')] / \
                                        bit[('gross', 'total')] - 1
            bit[('net', 'benefit')] = bit[('net', 'sum')] / bit[('net', 'total')] - 1
        bit = bit.sort_index(axis=0)
        bit = bit.sort_index(axis=1)
        bit = bit.iloc[[12, 13, 14, 15, 16, 17, 6,
                        7, 8, 9, 10, 11, 3, 5, 4, 2, 1, 0], :]

        pricing_summary = pd.concat(self.pricings.values(),
                                    keys=self.pricings.keys(),
                                    names=['portfolio', 'xx'],
                                    axis=1).droplevel(1, axis=1)

        self.audit_all = audit_all
        self.audit_var_tvar = bit
        self.pricing_summary = pricing_summary
        # return audit_all, bit, pricing_summary

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

    @classmethod
    def _display_work(cls, exhibit_id, df, caption, ff=None, save=True, cache_dir=None, show=False, align='right'):
        """
        Allow calling without creating an object, e.g. to consistently format the _cases_ database

        additional_properties is a list of selector, property pairs
        E.g. [(col-list, dict-of-properties)]

        :param exhibit_id:
        :param df:
        :param caption:
        :param ff:
        :param save:
        @return:
        """
        cell_hover = {  # for row hover use <tr> instead of <td>
            'selector': 'td:hover',
            'props': [('background-color', '#ffffb3')]
        }
        # color, matching color background fonts
        index_names = {
            'selector': '.index_name',
            'props': 'font-style: italic; color: black; background-color: #B4C3DC; font-weight:bold; border: 1px solid white; text-transform: capitalize; text-align:left;'
        }
        headers = {
            'selector': 'th:not(.index_name)',
            'props': 'background-color: #F1F8FE; color: black;  border: 1px solid #a4b3dc;'
        }
        center_heading = {
            'selector': 'th.col_heading',
            'props': 'text-align: center;'
        }
        left_index = {
            'selector': '.row_heading',
            'props': 'text-align: left;'
        }
        bold_level_0 = {
            'selector': 'th.col_heading.level0',
            'props': 'font-size: 1.2em;'
        }
        caption_f = {
            'selector': 'caption',
            'props': 'font-size: 1.1em; font-family: serif; font-weight: normal; text-align: left;'
        }
        td = {
            'selector': 'td',
            'props': f'text-align: {align}; border-bottom: 1px solid #a4b3dc;'
        }  # font-weight: bold;'}

        if exhibit_id != '':
            caption = f'({exhibit_id}) {caption}'

        # do the styling
        all_styles = [cell_hover, index_names, headers,
                      center_heading, bold_level_0, left_index, td, caption_f]
        styled_df = df.style.set_table_styles(all_styles)

        # if additional_properties is not None:
        #     for s, f in additional_properties:
        #         styled_df = styled_df.set_properties(s, **f)

        # if additional_styles is not None:
        #     styled_df = df.style.set_table_styles(additional_styles, overwrite=False)

        styled_df = styled_df.format(ff).set_caption(caption)

        if show is True and save is True:
            display(styled_df)
        if save:
            styled_df.to_html(buf=Path(cache_dir / f'{exhibit_id}.html'))
        process_memory()
        return styled_df

    def _display(self, exhibit_id, df, caption, ff=None, save=True):
        """

        caption = string caption
        ff = float format function

        """
        if ff is None:
            ff = self.default_float_format

        return self._display_work(exhibit_id, df, caption, ff, save, self.cache_dir, self.show)

    def show_exhibits(self, *chapters):
        if chapters[0] == 'all':
            chapters = [2, 4, 7, 9, 11, 13, 15]

        # ==================================================================================
        # ==================================================================================
        if 2 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 2: Case Study Intro</h2>'))
            ff = lambda x: f'{x:,.3f}'
            # table_no = [2.5, 2.6, 2.7][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""{self.case_name} estimated mean, CV, skewness and kurtosis by line and in total, gross and net.
{self.re_description} applied to {self.re_line}."""
            self._display("A", urn(self.audit_all.iloc[:4]), caption, ff)

        # ==================================================================================
        # ==================================================================================
        if 4 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 4: Risk Measures, VaR and TVaR</h2>'))
            ff = lambda x: f'{x:,.1f}' if abs(x) > 1 else f'{x:.3g}'
            # table_no = [4.6, 4.7, 4.8][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""{self.case_name} estimated VaR, TVaR and EPD by line and in total, gross and net.
EPD shows assets required for indicated EPD percentage.
Sum column shows sum of parts by line with no diversification and
benefit shows percentage reduction compared to total.
{self.re_description} applied to {self.re_line}."""
            self._display("E", urn(self.audit_var_tvar), caption, ff)

        # ==================================================================================
        # ==================================================================================
        if 7 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 7: Guide to Practice Chapters</h2>'))
            # table_no = 7.2
            # caption = f'Table {table_no}: '
            caption = f"""Pricing summary for {self.case_name} using a 
a {self.reg_p} capital standard and {self.roe:.1%} constant cost of capital for all layers."""
            self._display("F", urn(self.pricing_summary), caption, None)

            if self.show: display(HTML('<hr>'))
            # table_no = 7.3
            # caption = f'Table {table_no}: '
            # TODO come up with a replacement reinsurance using reinsurance_audit_df
            # caption = f"""Reinsurance summary for {self.case_name}."""
            # self._display("G", urn(self.re_summary), caption, None)

        # ==================================================================================
        # ==================================================================================
        if 9 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 9: Classical Portfolio Pricing</h2>'))
            # table_no = [9.2, 9.5, 9.8][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""Classical pricing by method for {self.case_name}. Pricing calibrated to total gross portfolio and applied to each line on a stand-alone basis.
Sorted by gross premium for {self.lns[1]}."""
            ff = lambda x: '' if np.isnan(x) else (f'{x:,.3f}' if abs(x) < 5 else f'{x:,.1f}')
            self._display("H", urn(self.classic_pricing), caption, ff)

            # table_no = [9.3, 9.6, 9.9][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""Sum of parts (SoP) stand-alone vs. diversified classical pricing by method for {self.case_name}. Delta
columns show the difference."""
            if self.show: display(HTML('<hr>'))
            self._display("I", urn(self.sop), caption, ff)

            if self.show: display(HTML('<hr>'))
            # table_no = [9.4, 9.7, 9.10][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""Implied loss ratios from classical pricing by method for {self.case_name}. Pricing calibrated to total gross portfolio and applied to each line on a stand-alone basis."""
            self._display("J", urn(self.lrs), caption, ff)

            if self.show: display(HTML('<hr>'))
            # table_no = 9.11
            # caption = f'Table {table_no}: '
            caption = f"""Comparison of stand-alone and sum of parts (SoP) premium for {self.case_name}."""
            tab911 = self.modern_monoline_sa.loc[
                         [('No Default', 'Loss'), ('No Default', 'Premium'), ('No Default', 'Capital'),
                          ('With Default', 'Loss'), ('With Default', 'Premium'), ('With Default', 'Capital')]].iloc[:,
                     [2, 3, 5, 6]]
            tab911.columns = ['Gross SoP', 'Gross Total', 'Net SoP', 'Net Total']
            tab911['Gross Redn'] = tab911['Gross Total'] / tab911['Gross SoP'] - 1
            tab911['Net Redn'] = tab911['Net Total'] / tab911['Net SoP'] - 1
            tab911 = tab911.iloc[:, [0, 1, 4, 2, 3, 5]]
            self._display("K", tab911, caption, lambda x: f'{x:,.1f}' if x > 10 else f'{x:.1%}')

            if self.show: display(HTML('<hr>'))
            # table_no = [9.12, 9.13, 9.14][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""Constant CoC pricing by unit for {self.case_name}, with {self.roe} cost of capital and p={self.reg_p}.
The column sop shows the sum by unit. {self.re_description} All units produce the same rate of return, by construction."""
            self._display("L", urn(self.modern_monoline_sa), caption, ff)  # lambda x: f'{x:.4g}')

        # ==================================================================================
        # ==================================================================================
        if 11 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 11: Modern Portfolio Pricing</h2>'))
            # table_no = 11.5
            # caption = f'Table {table_no}: '
            caption = f"""Parameter estimates for the five base spectral risk measures."""
            tab115 = self.gross.dist_ans.droplevel((0, 1), axis=0).loc[
                ['roe', 'ph', 'wang', 'dual', 'tvar'], ['param', 'error', '$P$', '$K$', 'ROE', '$S$']]. \
                rename(columns={'$P$': 'P', '$K$': 'K', '$S$': 'S', 'ROE': 'ι'})
            self._display("N", urn(tab115), caption, None)

            # table_no = [11.7, 11.8, 11.9][self.case_number]
            # caption = f'Table {table_no}: '
            if self.show: display(HTML('<hr>'))
            caption = f"""Traditional and stand alone Pricing by distortion.
Pricing by unit and distortion for {self.case_name}, calibrated to
CCoC pricing with {self.roe} cost of capital and $p={self.reg_p}$.
Losses and assets are the same for all distortions.
The column sop shows sum of parts by unit, the difference with the total
shows the impact of diversification.
{self.re_description}"""
            self._display("Q", urn(self.modern_monoline_by_distortion), caption, None)

        # ==================================================================================
        # ==================================================================================
        if 13 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 13: Classical Pricing Allocation</h2>'))
            # caption = 'Table 13.1: '
            caption = f"""Comparison of gross expected losses by line. Second column shows allocated 
recovery with total assets. Third column shows stand-alone limited expected value with stand-alone 
{self.reg_p}-VaR assets."""
            self._display("R", self.tab13_1, caption, None)

            # table_no = [13.2, 13.3, 13.4][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""Constant {self.roe:.2f} ROE pricing for {self.case_name}, classical PCP methods."""
            self._display("S", urn(self.classical_pricing), caption, None)

        # ==================================================================================
        # ==================================================================================
        if 15 in chapters:
            if self.show: display(HTML('<hr><h2>Chapter 15: Modern Pricing Allocation</h2>'))
            # table_no = [13.35, 13.36, 13.37][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""Constant {self.roe:.2f} ROE pricing for {self.case_name}, distortion, SRM methods."""
            self._display("V", urn(self.distortion_pricing), caption, lambda x: f'{x:.2f}' if abs(x) > 5 else f'{x:.3f}')

            if self.show: display(HTML('<hr>'))
            # table_no = [15.38, 15.39, 15.40][self.case_number]
            # caption = f'Table {table_no}: '
            caption = f"""{self.case_name} percentile layer of capital allocations compared to distortion allocations."""
            self._display("Y", urn(self.bodoff_compare), caption, lambda x: f'{x:.4g}')

    def show_extended_exhibits(self):
        """
        make the "extra" discrete and one-off exhibits

        handles creation (unlike exhibits....?)

        :param arvv:
        @return:
        """

        if self.f_discrete is True:
            bit = self.make_discrete()
            caption = 'Tables 11.1 and 11.2: computing expected loss and premium, PH 0.5 distortion.'
            self._display("Z-11-1", bit, caption, self.default_float_format2)

            bit = self.gross.augmented_df.query('loss==80').filter(regex='^(exag?_total)$').copy()
            bit.index.name = 'a'
            bit.columns = ['Limited Loss', 'Premium']
            caption = 'Table 11.3 outcomes with assets a=80 (answers only), PH 0.5 distortion.'
            self._display("Z-11-2", bit, caption, self.default_float_format2)

            bit = self.make_discrete('tvar')
            caption = 'Table 11.4 (Exercise): computing expected loss and premium, calibrated TVaR distortion.'
            self._display("Z-11-3", bit, caption, self.default_float_format2)

            bit = self.gross.augmented_df.query('loss==80').filter(regex='^(exag?_total)$').copy()
            bit.index.name = 'a'
            bit.columns = ['Limited Loss', 'Premium']
            caption = 'Table (new) outcomes with assets a=80 (answers only), TVaR distortion.'
            self._display("Z-11-4", bit, caption, self.default_float_format2)

        else:
            # progress of benefits - always show
            self.make_progression()
            caption = f'Table (new): Premium stand-alone by unit, sum, and total, and natural allocation by distortion.'
            self._display('Z-15-1', urn(self.progression.xs('Premium', axis=0, level=1)['gross']), caption)

            f1 = lambda x: f'{x:.3f}'
            fp = lambda x: f'{x:.0%}'
            fp1 = lambda x: f'{x:.1%}'
            ff = dict(zip(self.walk.columns, (f1, f1, f1, fp1, fp1, fp1, fp, fp, fp)))
            caption = f'Table (new): Progression of premium benefit by distortion. Differences between allocated and stand-alone, ' \
                      'the implied premium reduction, and the split by unit.'
            self._display('Z-15-2', self.walk, caption, ff)

            self.make_cap_table()
            caption = f'Table (new): Asset tranching with S&P bond default rates and a {self.d2tc:.1%} debt to ' \
                      'total capital limit.'
            # format by column name
            ff = {k: (lambda x: f'{x:.1%}') if k.find('Pct') >= 0
            else
            ((lambda x: f'{x:.4f}') if k.find('Pr') == 0
             else (lambda x: f'{x:.2f}'))
                  for k in self.cap_table.columns}
            self._display('Z-TR-1', self.cap_table, caption, ff);

    def make_cap_table(self, kind='gross'):
        """
        Suggest reasonable debt tranching for kind=(gross|net) subject to self.d2tc debt to total capital limit. 
        Uses S&P bond default analysis.
        
        This is a cluster. There must be a better way... 
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

        # return ans, tranches, bit, spdf, captable
        self.debt_stats = pd.Series(
            [a, el, premium, capital, equity, debt, debt_attach, prob_debt_attach, attach_rating, exhaust_rating],
            index=['a', 'EL', 'P', 'Capital', 'Equity', 'Debt', 'D_attach', 'Pr(D_attach)', 'Attach Rating',
                   'Exhaust Rating'])
        self.cap_table = cap_table
        self.sp_ratings = spdf

    def make_discrete(self, distortion_name='default'):
        """
        the extra exhibits for discrete cases (ch11)
        """
        if self.f_discrete == 0:
            logger.error('Ch 11 extended exhibis for discrete-based only.')
            return

        if distortion_name == 'default':
            d = agg.Distortion('ph', 0.5)
            self.gross.apply_distortion(d)
        else:
            self.gross.apply_distortion(self.gross.dists[distortion_name])

        bit = self.gross.augmented_df.query('p_total > 0').filter(regex='loss|p_t|S').copy()
        bit.index = range(1, len(bit) + 1)
        bit['X p'] = bit.loss * bit.p_total
        bit['X gp'] = bit.loss * bit.gp_total
        bit.columns = ['X', 'p', 'S', 'g(S)', 'Δg(S)', 'X p', 'X Δg(S)']
        bit = bit.iloc[:, [0, 1, 2, -2, 3, 4, -1]]
        bit.columns.name = 'j'

        bit['ΔX'] = bit.X.shift(-1) - bit.X
        bit['S ΔX'] = bit.S * bit['ΔX']
        bit['g(S) ΔX'] = bit['g(S)'] * bit['ΔX']

        bit.loc['Sum'] = bit.sum()
        bit.loc['Sum', ['X', 'S', 'g(S)']] = np.nan

        bit = bit[['X', 'ΔX', 'p', 'S', 'X p', 'S ΔX', 'g(S)', 'Δg(S)', 'X Δg(S)', 'g(S) ΔX']]
        return bit

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
        S = (1 - df.p_total[0:a].cumsum())
        bs = self.gross.bs

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
            temp = pd.Series(distortion.g(S)[::-1], index=S.index)
            temp = temp.shift(1, fill_value=0).cumsum() * bs
            return temp.iloc[-1]

        # two methods of calibration
        if self.f_blend_extend == 0:

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

        elif self.f_blend_extend == 1:
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

    def get_f_axs(self, nr, nc):
        w = nc * self.fw
        h = nr * self.fh
        return self.smfig(nr, nc, (w, h))

    def apply_distortions(self, dnet_='', dgross_=''):
        """
        make the 12 up plot
        """

        # if nothing is passed in
        dgross = 'wang'
        dnet = 'wang'

        # book default distortions
        if self.case_id.startswith('discrete'):
            dnet = 'tvar'
            dgross = 'roe'
        elif self.case_id.startswith('tame'):
            dnet = 'tvar'
            dgross = 'roe'
        elif self.case_id.startswith('cnc'):
            dnet = 'ph'
            dgross = 'dual'
            # dnet = 'roe'
            # dgross = 'roe'
        elif self.case_id.startswith('hs'):
            dnet = 'wang'
            dgross = 'blend'

        # do what's asked for
        if dnet_ != '':
            dnet = dnet_
        if dgross_ != '':
            dgross = dgross_
        self.dnet = dnet
        self.dgross = dgross

        net = self.ports['net']
        net.apply_distortion(net.dists[dnet])
        gross = self.ports['gross']
        gross.apply_distortion(gross.dists[dgross])

    def _display_plot(self, plot_id, f, caption):
        """
        Save the graphic
        """
        # save the graph
        p = self.cache_dir / f'{plot_id}.{self.figtype}'
        f.savefig(p, dpi=600)
        if self.show is False:
            plt.close(f)

        # make the container html snippet
        pth = str(p.relative_to(self.cache_base).as_posix())
        blob = f"""
<figure>
<img src="/{pth}" width="100%" alt="Figure {f}" style="width:{100}%">
<figcaption class="caption">{caption}</figcaption>
</figure>
"""
        (self.cache_dir / f'{plot_id}.html').write_text(blob, encoding='utf-8')
        process_memory()

    def twelve_plot(self):
        """
        make the 12 up plot
        must call apply_distortions first!
        """
        # variables
        if self.case_id.startswith('discrete'):
            ylim = [-0.00025 / 4, .0075 / 4]
            xlim = [-5, 105]
            sort_order = [0, 1, 2]
            multiple_locator = .5e2
            legend_loc = 'center right'
        elif self.case_id.startswith('tame'):
            ylim = [-0.00025 / 4, .0075 / 4]
            xlim = [-20, 175]
            sort_order = [2, 0, 1]
            multiple_locator = 1e2
            legend_loc = 'lower left'
        elif self.case_id.startswith('cnc'):
            ylim = [-0.00025 / 4, .0075 / 4]
            xlim = [-20, 300]
            sort_order = [2, 0, 1]
            multiple_locator = 1e2
            legend_loc = 'lower left'
        elif self.case_id.startswith('hs'):
            ylim = [-0.00025 / 4, .01 / 4]
            xlim = [-20, 2000]
            sort_order = [2, 0, 1]
            multiple_locator = 5e2
            legend_loc = 'center right'
        else:
            # initial version...
            ylim = self.gross.limits(stat='density')
            xlim = self.gross.limits()
            xm = round_bucket(xlim[1])
            xlim = [-xm / 50 , xm]
            sort_order = [2, 0, 1]
            multiple_locator = round_bucket(xm / 10)
            legend_loc = 'center right'
        # net / gross flavors 15.2-3
        captions = [
            f'(TN) {self.case_name}, net twelve plot with {self.dnet} distortion.',
            f'(TG) {self.case_name}, gross twelve plot with {self.dgross} distortion.'
        ]
        for port, nm, caption in zip((self.net, self.gross), ('T_net', 'T_gross'), captions):
            f, axs = self.smfig(4, 3, (10.8, 12.0))
            twelve_plot(port, f, axs, xmax=xlim[1], ymax2=xlim[1],
                        contour_scale=port.q(self.reg_p), sort_order=sort_order, cmap=self.colormap)
            axs[0][0].set(ylim=ylim, xlim=xlim)
            axs[0][1].set(xlim=xlim)
            axs[0][2].yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator))
            axs[1][0].yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator))
            axs[2][2].yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator))
            axs[3][2].yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator))
            axs[2][2].legend().set(visible=False)
            axs[3][2].legend().set(visible=False)
            if self.case_id == 'hs':
                axs[3][1].set(ylim=[-10, 200])
            axs[1][2].legend(loc=legend_loc)
            self._display_plot(nm, f, caption)

    def show_extended_graphs(self):
        """
        Create relevant extended graphs

        @return:
        """
        if self.f_discrete:
            return
        # graph of tranching
        f, axs = self.smfig(1, 2, (12.0, 4.0))
        ax0, ax1 = axs.flat
        port = self.gross
        el = self.debt_stats['EL']
        prem = self.debt_stats['P']
        cap = self.debt_stats['a']
        max_debt = self.debt_stats['D_attach']

        # just show the full letter levels
        ix = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
        for ax in axs.flat:
            self.gross.density_df.p_total.plot(ax=ax, lw=1.5, label='Loss density')
            for n, x in self.sp_ratings.loc[ix].iterrows():
                ax.axvline(x.attachment, c='C7', lw=.25)
            ax.axvline(el, c='C1', lw=1.5, label=f'EL={el:,.0f} @ {port.sf(el):.3%}')
            ax.axvline(prem, c='C2', lw=1.5, label=f'Premium={prem:,.0f} @ {port.sf(prem):.3%}')
            ax.axvline(max_debt, c='C3', lw=1.5, label=f'Debt attachment={max_debt:,.0f} @ {port.sf(max_debt):.3%}')
            msg = f'(max debt={cap - max_debt:.0f}, capital={cap - prem:,.0f})'
            ax.axvline(cap, c='C4', lw=1.5, label=f'Assets={cap:,.0f} @ {port.sf(cap):.3%},\n{msg}')
            ax.xaxis.set_major_locator(ticker.FixedLocator(self.sp_ratings.loc[ix, 'attachment']))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{i}\n{self.sp_ratings.loc[i, "attachment"]:,.0f}\n'
                                                                f'{self.sp_ratings.loc[i, "default"]:.3%}' for i in
                                                                ix]))

        xmin = self.gross.q(0.0001)
        xmax = self.gross.q(1 - .00003 / 2)
        if self.case_id == 'tame':
            xmax *= 1.25
        ax0.set(xlabel='Loss', ylabel='Density')
        ax0.set(xlim=[xmin, xmax], yscale='linear')
        ax1.set(xlim=[xmin, xmax], ylim=1e-10, yscale='log', xlabel='Loss')
        ax0.legend(loc='upper right')
        caption = f'(Z-TR-1) Figure (new): capital tranching with a {self.d2tc:.1%} debt to total capital limit.'
        self._display_plot('Z-TR-1', f, caption)

        f, axs = self.smfig(1, 3, (12.0, 4.0), )
        ax0, ax1, ax2 = axs.flat
        d0 = self.make_blend()
        dd = self.bond_pricing_table()

        ps = np.hstack((np.linspace(0, 0.1, 200000, endpoint=False), np.linspace(0.1, 1, 1000)))
        droe = agg.Distortion('roe', self.roe).g(ps)
        blend = d0.g(ps)
        fit_blend = self.blend_d.g(ps)
        ph = self.gross.dists['ph'].g(ps)

        for ax in axs.flat:
            ax.plot(ps, ps, c='k', lw=0.5)
            ax.plot(ps, droe, label='CCoC')
            ax.plot(ps, blend, label='Naive blend')
            ax.plot(ps, fit_blend, label='Calibrated blend')
            ax.plot(dd.default, dd['yield'], 'x', lw=.5, label='Bond pricing data')
            ax.plot(ps, ph, lw=0.5, label='PH')
            ax.legend()
        ax1.set(xlim=[0, 0.1], ylim=[0, 0.1])
        ax2.set(xscale='log', yscale='log', xlim=[.8e-6, 1], ylim=[.8e-6, 1])

        caption = f'(Z-BL-1) Figure (new): Calibrated blend distortion, compard to CCoC and naive blend. The x marks show bond default and pricing data used in calibration. ' \
                  'Middle plot zooms into 0 < s < 0.1. Righthand plot uses a log/log scale. PH distortion added for comparison.'
        self._display_plot('Z-BL-1', f, caption)

    def show_similar_risks_graphs(self, base='gross', new='net'):
        """
        axd from mosaic
        Treats the bounds correctly in computing the tvars for the new portfolio

        Provenance : from make_port in Examples_2022_post_publish, similar_risks_graphs_sa

        """
        bounds = self.boundss[base]
        df = bounds.weight_df.copy()
        df['test'] = df['t_upper'] * df.weight + df.t_lower * (1 - df.weight)

        # tvars for the new risk
        new_a = self.pricing_summary.at['a', new]
        new_prem = self.pricing_summary.at['P', new]
        tvarf = self.boundss[new].tvar_with_bound
        tvar1 = {p: float(tvarf(p, new_a)) for p in bounds.tps}
        df['t1_lower'] = [tvar1[p] for p in df.index.get_level_values(0)]
        df['t1_upper'] = [tvar1[p] for p in df.index.get_level_values(1)]
        df['t1'] = df.t1_upper * df.weight + df.t1_lower * (1 - df.weight)

        a = self.pricing_summary.at['a', base]
        prem = self.pricing_summary.at['P', base]
        roe_d = agg.Distortion('roe', self.roe)
        tvar_d = agg.Distortion('tvar', bounds.p_star('total', prem))
        idx = df.index.get_locs(df.idxmax()['t1'])[0]
        pl, pu, tl, tu, w = df.reset_index().iloc[idx, :-4]
        max_d = agg.Distortion('wtdtvar', w, df=[pl, pu])

        tmax = float(df.iloc[idx]['t1'])
        print('Ties for max: ', len(df.query('t1 == @tmax')))
        print('Near ties for max: ', len(df.query('t1 >= @tmax - 1e-4')))

        idn = df.index.get_locs(df.idxmin()['t1'])[0]
        pln, pun, tl, tu, wn = df.reset_index().iloc[idn, :-4]
        min_d = agg.Distortion('wtdtvar', wn, df=[pln, pun])

        # make the plot mosaic
        f = plt.figure(constrained_layout=True, figsize=(16, 8))
        axd = f.subplot_mosaic(
            """
            AAAABBFF
            AAAACCFF
            AAAADDEE
            AAAADDEE
        """
        )

        ax = axd['A']
        plot_max_min(bounds, ax)
        n = len(ax.lines)
        roe_d.plot(ax=ax, both=False)
        tvar_d.plot(ax=ax, both=False)
        max_d.plot(ax=ax, both=False)
        min_d.plot(ax=ax, both=False)

        ax.lines[n + 0].set(label='roe')
        ax.lines[n + 2].set(color='green', label='tvar')
        ax.lines[n + 4].set(color='red', label='max')
        ax.lines[n + 6].set(color='purple', label='min')
        ax.legend(loc='upper left')

        ax.set(title=f'Max ({pl:.4f}, {pu:.4f}), min ({pln:.4f}, {pun:.4f})')

        ax = axd['B']
        bounds.weight_image(ax)
        ax.set(title=f'Weight for p1 on {base}')

        bit = df['t1'].unstack(1)
        ax = axd['C']
        img = ax.contourf(bit.columns, bit.index, bit, cmap='viridis_r', levels=20)
        ax.set(xlabel='p1', ylabel='p0', title=f'Pricing on {new} Risk', aspect='equal')
        ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16, label='rho(X_new)')
        ax.plot(pu, pl, '.', c='w')
        ax.plot(pun, pln, 's', ms=3, c='white')

        ax = axd['D']
        port = self.ports[base]
        pnew = self.ports[new]

        def plot_lee(port, ax, lw=2, scale='linear'):
            """
            Lee diagram by hand
            """
            p_ = np.linspace(0, 1, 10000, endpoint=False)
            qs = [port.q(p) for p in p_]
            if scale == 'linear':
                ax.step(p_, qs, lw=lw, label=port.name)
                ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, port.q(0.9999) + .05])
            else:
                ax.plot(1 / (1 - p_), qs, lw=lw, label=port.name)
                ax.set(xscale='log', xlim=[0.5, 2e4], ylim=[-0.05, port.q(0.9999) + .05])

        plot_lee(pnew, ax, scale='log')
        plot_lee(port, ax, lw=1, scale='log')
        ax.set(title=f'Lee Diagrams - log return')
        ax.legend(loc='upper left')

        ax = axd['E']
        plot_lee(pnew, ax)
        plot_lee(port, ax, lw=1)
        ax.set(title=f'Lee Diagrams - linear')
        ax.legend(loc='upper left')

        # port.density_df.p_total.plot(ax=ax, label=port.name)
        # pnew.density_df.p_total.plot(ax=ax, label=pnew.name)
        # ax.set(xlim=[0, pnew.q(0.9999)], title='Densities')
        # ax.legend(loc='upper right')

        ax = axd['F']
        plot_max_min(bounds, ax)
        for c, dd in zip(['r', 'g', 'b'], ['ph', 'wang', 'dual']):
            port.dists[dd].plot(ax=ax, both=False, lw=1)
            ax.lines[n].set(c=c, label=dd)
            n += 2
        ax.set(title='PH, Wang and Dual')
        ax.legend(loc='lower right')

        caption = f'(Z-TR-2) Figure (new): Analysis of extreme distortions for net portfolio based on gross calibration. ' \
                  'Large plot shows the extreme distortions achieving the greatest and least premium for the net portfolio.'
        self._display_plot('Z-TR-2', f, caption)

        return df

    def loss_density_spectrum(self):
        """
        Loss (exeqa) by line and the distortion spectrums.

        This is different for discrete distributions.

        :param self:
        @return:
        """

        logger.info("15.11")
        f, axs = self.get_f_axs(3, 2)

        def diff(f):
            """ manual differentiation """
            eps = 1e-9

            def fprime(x):
                return (f(x + eps) - f(x - eps)) / (2 * eps)

            return fprime

        if self.case_id.startswith('discrete'):
            self.exeqa = {}
            for port in [self.gross, self.net]:
                bit = port.density_df.query('p_total > 0').filter(regex='exeqa_[tX]|F').set_index('F', drop=True)
                bit.loc[0] = 0.
                bit = bit.sort_index().rename(columns=lambda x: x.split('_')[1].title())
                self.exeqa[port.name] = bit

            ps = bit.index
            gprime = {}
            dist_list = ['roe', 'ph', 'wang', 'dual', 'tvar', 'blend']
            for dn in dist_list:
                gprime[dn] = diff(self.gross.dists[dn].g)

            self.diff_g = pd.DataFrame({dn: gprime[dn](1 - ps) for dn in dist_list},
                                       index=bit.index)

            names = {'roe': 'CCoC', 'ph': 'Prop Hazard', 'wang': 'Wang',
                     'dual': "Dual", 'tvar': "TVaR", 'blend': "Blend"}

            lbl = 'Gross E[Xi | X]'
            ax0 = axs[0, 0]
            for (k, b), ax in zip(self.exeqa.items(), axs.flat):
                b.plot(drawstyle='steps-pre', ax=ax)
                ax.set(xlim=[-0.025, 1.025], ylabel=lbl, xlabel='p')
                ax.lines[0].set(lw=3, alpha=.5)
                ax.axhline(0, lw=.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(.25))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1 / 16))
                # gross and net have same y lim
                if ax is ax0:
                    ylim_ = ax.get_ylim()
                else:
                    ax.set(ylim=ylim_)
                lbl = 'Net E[Xi | X]'
                ax.legend(loc='upper left')

            axi = iter(axs.flat[2:])
            ylim = 5
            for d1, d2 in zip(*[iter(dist_list)] * 2):
                ax = next(axi)
                # removed c='k'
                ax.plot(self.diff_g.index, self.diff_g[d1], c='C2', lw=1, label=names[d1], drawstyle='steps-post')
                if d1 == 'roe':
                    ax.plot([1], [ylim - 0.1], '^', c='C2', ms=10, label='CCoC mass')
                ax.plot(self.diff_g.index, self.diff_g[d2], c='C3', lw=1, label=names[d2], drawstyle='steps-post')
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(.25))
                ax.axhline(1, lw=.5, c='k')
                ax.set(xlim=[-0.025, 1.025], ylim=[0, ylim], xlabel=None,
                       ylabel='Distortion weight (spectrum)')
                ax.legend(loc='upper left')

            axs[2, 0].set(xlabel='Loss non-exceedance probability')
            axs[1, 1].set(ylabel=None)
            axs[2, 1].set(ylabel=None)

            # plot them all together
            ax = axs[2, 1]
            # omit blend
            self.diff_g.iloc[:, :-1].plot(ax=ax, lw=1, drawstyle='steps-pre')
            ax.set(xlim=[-0.025, 1.025], yscale='linear', xlabel='Loss non-exceedance probability')
            ax.axhline(1, lw=.5, c='k')

        else:
            self.exeqa = {}
            for port in [self.gross, self.net]:
                bit = port.density_df.filter(regex='loss|^S$|^F$|exeqa_[A-Zt]')
                if self.case_id == 'hs':
                    bit = bit.iloc[::8]
                bit['return'] = 1 / bit.S
                bit0 = bit.set_index('return').filter(regex='ex').rename(columns=lambda x: x.split('_')[1]).drop(
                    'total', axis=1).cumsum(axis=1)
                bit0 = bit0.loc[bit0.index < np.inf]

                ns = 1000
                if self.case_id == 'hs':
                    ns = 10000
                qs = pd.qcut(bit0.index, ns, duplicates='drop')
                bit0['qs'] = qs
                bit0 = bit0.reset_index(drop=False).groupby(qs).agg(np.mean).set_index('return')
                self.exeqa[port.name] = bit0

            ps = 1 / bit0.index
            gprime = {}
            dist_list = ['roe', 'ph', 'wang', 'dual', 'tvar', 'blend']
            for dn in dist_list:
                gprime[dn] = diff(self.gross.dists[dn].g)

            self.diff_g = pd.DataFrame({dn: gprime[dn](ps) for dn in dist_list},
                                       index=bit0.index)

            names = {'roe': 'CCoC', 'ph': 'Prop Hazard', 'wang': 'Wang',
                     'dual': "Dual", 'tvar': "TVaR", 'blend': "Blend"}

            lbl = 'Gross E[Xi | X]'
            for (k, b), ax in zip(self.exeqa.items(), axs.flat):
                b.plot(ax=ax)
                ylim = {'tame': 200, 'cnc': 500, 'hs': 2500}.get(self.case_id, self.gross.q(0.999))
                ax.set(xlim=[1, .5e6], ylim=[0, ylim], xscale='log', ylabel=lbl, xlabel=None)
                lbl = 'Net E[Xi | X]'
                ax.legend(loc='upper left')

            axi = iter(axs.flat[2:])
            ylim = 5
            if k == 'hs':
                ylim = 50
            for d1, d2 in zip(*[iter(dist_list)] * 2):
                ax = next(axi)
                # removed c='k'
                ax.plot(self.diff_g.index, self.diff_g[d1], c='C2', lw=1, label=names[d1])
                if d1 == 'roe':
                    ax.plot([.4e5], [ylim - 0.1], '^', c='C2', ms=10, label='CCoC mass')
                ax.plot(self.diff_g.index, self.diff_g[d2], c='C3', lw=1, label=names[d2])
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ylim / 5))
                ax.axhline(1, lw=.5, c='k')
                ax.set(xlim=[0.5, .5e5], ylim=[0, ylim], xscale='log', xlabel=None,
                       ylabel='Distortion weight (spectrum)')
                ax.legend(loc='upper left')

            axs[2, 0].set(xlabel='Loss return period')
            axs[1, 1].set(ylabel=None)
            axs[2, 1].set(ylabel=None)

            # plot them all together
            ax = axs[2, 1]
            # omit blend
            self.diff_g.iloc[:, :-1].plot(ax=ax, lw=1)
            ax.set(xlim=[0.5, 0.5e5], ylim=[.1, 200], xscale='log', yscale='log', xlabel='Loss return period')

        caption = f'(W) Figure 15.11: {self.case_name}, loss spectrum (gross/net top row). Rows 2 and show VaR weights by distortion. In the second row, the CCoC distortion includes a mass putting weight 𝑑 = {self.roe}∕{1 + self.roe} at the maximum loss, corresponding to an infinite density. ' \
                  'The lower right-hand plot compares all five distortions on a log-log scale.'
        self._display_plot('W', f, caption)

    def show_graphs(self, *chapters):
        if chapters[0] == 'all':
            chapters = [2, 4, 7, 9, 11, 13, 15]

        # ==================================================================================================
        # ==================================================================================================
        if 2 in chapters:
            # fig 2.2, 2.4, and 2.6
            logger.info('Figures 2.2, 2.4, 2.6')
            # f, axs = self.get_f_axs(2, 2)
            # ax0, ax1, ax2, ax3 = axs.flat

            f, axd = make_mosaic_figure('AB\nCD')
            axd_top = {'A': axd['A'], 'B': axd['B']}
            axd_bottom = {'A': axd['C'], 'B': axd['D']}
            # order series so total is first
            self.gross.plot(axd_top)
            self.net.plot(axd_bottom)
            caption = f'(B) {self.case_name}, gross (top) and net (bottom) densities on a nominal (left) and log (right) scale.'
            self._display_plot('B', f, caption)

            # # limits for graphs
            # if self.case_id == 'cnc':
            #     regex = 'p_[CNt]'
            #     a = 300  # gross.q(reg_p)
            # elif self.case_id == 'tame':
            #     regex = 'p_[ABt]'
            #     a = 200
            # elif self.case_id == 'hs':
            #     regex = 'p_[HSt]'
            #     a = 1000
            # elif self.case_id == 'discrete':
            #     regex = 'p_[Xt]'
            #     a = 110
            # else:
            #     # lines should be called A and B
            #     regex = f'p_({self.a_name}|{self.b_name}|total)'
            #     if self.f_discrete:
            #         a = self.gross.q(1)
            #     else:
            #         a = self.gross.q((1 + self.reg_p) / 2)
            #
            # if self.f_discrete:
            #     i = 0
            #     for ax, ln in zip(axs.flat, self.gross.line_names_ex):
            #         self.gross.density_df[f'p_{ln}'].cumsum().plot(drawstyle='steps-post', lw=2, ax=ax, c=f'C{i}',
            #                                                        label=f'Gross {ln}')
            #         if ln == self.re_line or ln == 'total':
            #             self.net.density_df[f'p_{ln}'].cumsum().plot(drawstyle='steps-post', ls='--', lw=1, ax=ax,
            #                                                          c=f'C{i}', label=f'Net {ln}')
            #         ax.set(xlim=[-1, a + 1], ylim=[-0.025, 1.025])
            #         ax.legend(loc='lower right')
            #         if self.case_id == 'discrete':
            #             ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            #         i += 1
            #
            #     ax = axs.flatten()[-1]
            #     self.gross.density_df.filter(regex='p_[tX]').cumsum().plot(drawstyle='steps-post', ax=ax)
            #     ax.set(xlim=[-1, a + 1], ylim=[-0.025, 1.025])
            #     ax.legend(loc='lower right')
            #     if self.case_id == 'discrete':
            #         ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            #     caption = f'(B) Figure (suppl.) 2.2: {self.case_name}, gross and net densities by line and combined gross.',
            #     self._display_plot('B', f, caption)
            #
            # else:
            #     self.gross.density_df.filter(regex=regex).iloc[:, [2, 0, 1]].rename(
            #         columns=lambda x: x.replace('p_', 'Gross ')).plot(ax=ax0)
            #     self.gross.density_df.filter(regex=regex).iloc[:, [2, 0, 1]].rename(
            #         columns=lambda x: x.replace('p_', 'Gross ')).plot(ax=ax1, logy=True)
            #
            #     self.net.density_df.filter(regex=regex).iloc[:, [2, 0, 1]].rename(
            #         columns=lambda x: x.replace('p_', 'Net ')).plot(ax=ax2)
            #     self.net.density_df.filter(regex=regex).iloc[:, [2, 0, 1]].rename(
            #         columns=lambda x: x.replace('p_', 'Net ')).plot(ax=ax3, logy=True)
            #
            #     for ax in axs.flat:
            #         ax.set(xlim=[-a / 50, a])
            #
            #     if ax in [ax1, ax3]:
            #         ax.legend().set(visible=False)
            #
            #     if self.case_id in ['tame', 'cnc']:
            #         ax2.set(ylim=[-0.00125 / 16, .025 / 16])
            #     else:
            #         ax0.set(ylim=[-0.0025 / 2, .025])
            #         ax2.set(ylim=[-0.005, .1])


            # fig 2.3, 2.5, and 2.6  ===================================================================================
            logger.info('Figures 2.3, 2.5, 2.7')

            if self.case_id == 'cnc':
                xmax = 500
                sim_xlim = [-10, 500]
                sim_ylim = [-10, 500]
                sim_scale = 250
            elif self.case_id == 'tame':
                xmax = 250 / 2
                sim_xlim = [-5, 250 / 2]
                sim_ylim = [-5, 250 / 2]
                sim_scale = 125 / 2
            elif self.case_id == 'hs':
                xmax = 2000
                sim_xlim = [-10, 2000]
                sim_ylim = [-10, 2000]
                sim_scale = 2000
            elif self.case_id == 'discrete':
                xmax = 110
                sim_xlim = [-5, 110]
                sim_ylim = [-5, 110]
                sim_scale = 100
            else:
                logger.warning('Check scales are reasonable...')
                sim_xlim = self.gross.limits()
                xmax = round_bucket(sim_xlim[1])
                sim_xlim = [-xmax / 50, xmax]
                sim_ylim = sim_xlim
                sim_scale = xmax // 4  # ? QDFC

            # need to make this work for the other version
            ln1, ln0 = self.gross.line_names
            pln0 = f'p_{ln0}'
            pln1 = f'p_{ln1}'
            port = self.gross
            sample = pd.concat((port.density_df[['loss', pln0]].sample(10 ** 6, replace=True, weights=pln0).reset_index(
                drop=True).rename(columns={'loss': ln0}),
                                port.density_df[['loss', pln1]].sample(10 ** 6, replace=True, weights=pln1).reset_index(
                                    drop=True).rename(columns={'loss': ln1})),
                               axis=1)
            sample['loss'] = sample[ln1] + sample[ln0]
            sample = sample.set_index('loss').sort_index()
            sample['e_total'] = sample[ln0] / sample.index
            sample['bin'] = pd.qcut(sample.index, 1000, duplicates='drop')
            # kappa_estimate = sample.reset_index().groupby('bin')[['loss', 'e_total']].agg(np.mean).set_index('loss')

            # make the image
            f, ax2 = self.smfig(1, 1, (3, 3))
            bit = sample.sample(n=25000 if self.case_id == 'tame' else 250000)
            ax2.scatter(x=bit[ln1], y=bit[ln0], marker='.', s=1, alpha=.1)
            ax2.set(xlim=sim_xlim, ylim=sim_ylim)
            ax2.set(aspect='equal')
            ax2.axis('off')
            bit_fn = Path.home() / f'aggregate/temp/scatter_{self.case_id}.png'
            f.savefig(bit_fn, dpi=1000)
            plt.close(f)
            del f

            # assemble parts
            f, axs = self.smfig(1, 3, (9, 3))
            ax0, ax1, ax2 = axs.flat
            axi = iter(axs.flat)
            bivariate_density_plots(axi, self.ports.values(), xmax=xmax, levels=15, biv_log=True, contour_scale=xmax,
                                    cmap=self.colormap)

            img = plt.imread(bit_fn)
            ax2.imshow(img, extent=[*sim_xlim, *sim_ylim])

            ax2.set(xlim=sim_xlim, ylim=sim_ylim, )
            tvar = self.gross.q(.99)
            ax2.plot([tvar, 0], [0, tvar], 'k', lw=1)
            for ax in axs.flat:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(sim_scale))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(sim_scale))
            ax2.set(aspect='equal', title=self.gross.name + ' sample')
            ax0.set(title='Gross')
            ax1.set(title='Net', ylabel=None)
            x, y = self.ports['gross'].line_names
            ax2.set(title='Gross sample', xlabel=f'Line {x}')
            caption = f'(C) {self.case_name}, bivariate densities: gross (left), net (center), and a sample from gross (right). Impact of reinsurance is clear in net plot.'
            self._display_plot('C', f, caption)

        # ==================================================================================================
        # ==================================================================================================
        if 4 in chapters:
            # fig 4.10, 11, 12 ===================================================================================
            logger.info('Figures 4.10, 4.11, 4.12')
            f, axs = self.get_f_axs(2, 2)
            # ax0, ax1, ax2, ax3 = axs.flat

            for (k, port), ax0, ax1 in zip(self.ports.items(), axs[0], axs[1]):
                bounds = self.boundss[k]
                a = self.pricing_summary.at['a', k]

                if self.case_id not in ('discrete', 'awkward', 'differentialble', 'notdifferentiable'):
                    df = pd.DataFrame({'p': bounds.tps,
                                       'TVaR': [float(port.tvar(i)) for i in bounds.tps],
                                       'Bounded TVaR': bounds.tvars,
                                       'VaR': [port.q(i) for i in bounds.tps]}).set_index('p')
                    pu = df.index[-2]
                    for i in range(3):
                        pu = (1 + pu) / 2
                        tvarb = bounds.tvar_with_bound(pu, a)
                        tvar = bounds.tvar_function(pu)
                        q = port.q(pu)
                        df.loc[pu, :] = [tvar, tvarb, q]
                    df = df.sort_index()
                    ds = 'default'
                else:
                    # for discrete need more careful, non-interpolating version of tvar
                    df = pd.DataFrame({'p': bounds.tps,
                                       'TVaR': [bounds.tvar_with_bound(i, a, 'tail') for i in bounds.tps],
                                       'VaR': [port.q(i) for i in bounds.tps]
                                       }).set_index('p')
                    ds = 'steps-pre'

                df['Bounded VaR'] = np.minimum(df.VaR, a)
                ax = ax0
                df.plot(ax=ax, drawstyle=ds)
                # ax.lines[-1].set(lw=3, ls=':', alpha=.5)
                ax0.set(xlabel='$p$')
                if self.case_id == 'discrete':
                    ax.set(ylim=[0, 110])
                else:
                    ax.set(ylim=[port.q(0.001), port.q(max(0.9999, self.reg_p))])
                ax.set(title=f'{k.title()}, a={a:,.1f}')
                ax.legend(loc='upper left')

                ax = ax1
                df.index = 1 / (1 - df.index)
                df.plot(ax=ax, drawstyle=ds)
                # ax.lines[-1].set(lw=3, ls=':', alpha=.5)
                if self.case_id == 'discrete':
                    ax.set(ylim=[0, 110])
                else:
                    pass
                    # ax.set(ylim=[0, 500]) # port.q(max(0.9999, reg_p))])
                ax.set(xlabel='Return period', xscale='log', title='Return period view')
                ax.legend().set(visible=False)

            caption = f'(D) Figure 4.10: {self.case_name}, TVaR, and VaR for unlimited and limited variables, gross (left) and net (right). Lower view uses a log return period horizontal axis.'
            self._display_plot('D', f, caption)

        # ==================================================================================================
        # ==================================================================================================
        if 11 in chapters or 11.3 in chapters:
            # fig 11.3, 4, 5
            # warning: quite slow!
            logger.info('Figures 11.3, 11.4, 11.5')
            f, axs = self.smfig(1, 3, (3 * 3, 1 * 3))
            bounds = self.boundss['gross']
            port = self.ports['gross']
            premium, a = self.pricing_summary.loc[['P', 'a'], 'gross']
            p_star = bounds.p_star('total', premium, a)
            bounds.cloud_view(axs.flatten(), 0, alpha=1, pricing=True,
                              title=f'Premium={premium:,.1f}, p={self.reg_p:.1g}\na={a:,.0f}, p*={p_star:.3f}',
                              distortions=[{k: port.dists[k] for k in ['roe', 'tvar']},
                                           {k: port.dists[k] for k in ['ph', 'wang', 'dual']}])
            # for ax in axs.flat:
            #     ax.set(aspect='equal')
            # f.suptitle(f'Portfolio {k}', fontsize='x-large')
            for ax in axs.flatten()[1:]:
                ax.legend(ncol=1, loc='lower right')
            for ax in axs.flatten():
                ax.set(title=None)
            caption = f'(M) Distortion envelope for {self.case_name}, gross. Left plot shows the distortion envelope, middle overlays the CCoC and TVaR distortions, right overlays proportional hazard, Wang, and dual moment distortions.'
            self._display_plot('M', f, caption)

        if 11 in chapters or 11.6 in chapters:
            # fig 11.6, 7, 8
            logger.info('Figures 11.6, 11.7, 11.8')
            f, axs = self.smfig(4, 3, (3 * 2.5, 4 * 2.5))

            port = self.ports['gross']
            for dn, ls in zip(['ph', 'wang', 'dual'], ['-', '--', ':']):
                axi = iter(axs.flatten())
                g_ins_stats(axi, dn, port.dists[dn], ls=ls)

            for dn, ls in zip(['roe', 'tvar', 'blend'], ['-', '--', ':']):
                axi = iter(axs.flatten()[6:])
                g_ins_stats(axi, dn, port.dists[dn], ls=ls)

            caption = f'(O) {self.case_name}, variation in premium, loss ratio, markup (premium to loss), margin, discount rate, and premium to capital leverage for six distortions, shown in two groups of three. Top six plots show proportional hazard, Wang, and dual moment; lower six: CCoC, TVaR, and Blend.'
            self._display_plot('O', f, caption)

        if 11 in chapters or 11.9 in chapters:
            # fig 11.9, 10, 11
            logger.info('Figures 11.9, 11.10, 11.11')
            f, axs = self.smfig(6, 4, (4 * 2.5, 6 * 2.5))

            port = self.ports['gross']
            for i, dn in zip(range(6), ['roe', 'ph', 'wang', 'dual', 'tvar', 'blend']):
                axi = iter(axs.flatten()[4 * i:])
                macro_market_graphs(axi, port, dn, 200)
                if i > 0:
                    axi = iter(axs.flatten()[4 * i:])
                    next(axi).legend().set(visible=False)
                    next(axi).legend().set(visible=False)
                    next(axi).legend().set(visible=False)
                    next(axi).legend().set(visible=False)

            caption = f'(P) {self.case_name}, variation in SRM properties as the asset limit (x-axis) is varied. Column 1: total premium and loss; 2: total assets, premium, and capital; 3; total and layer loss ratio; and 4: total and layer discount factor. ' \
                      'By row CCoC, PH, Wang, Dual, TVaR, and Blend.'
            self._display_plot('P', f, caption)

        # ==================================================================================================
        # ==================================================================================================
        if 15 in chapters:

            # fig 15.2--15.7 (Twelve Plot)
            self.twelve_plot()

            # fig 15.8, 15.9, 15.10
            logger.info("Figures 15.8, 15.9, 15.10")
            f, axs = self.get_f_axs(1, 2)
            ax0, ax1 = axs.flat
            if self.case_id == 'tame':
                xlim = [1, 1e9]
            elif self.case_id == 'cnc':
                xlim = [1, 1e5]
            elif self.case_id == 'hs':
                xlim = [1, 1e5]
            elif self.case_id == 'discrete':
                xlim = [0, 100]
            else:
                # TO DO! think about this!
                xlim = self.gross.limits()

            for port, kind, ax in zip([self.gross, self.net], ['Gross', 'Net'], axs.flat):
                bit = port.augmented_df.filter(regex='M\.Q|F|^S$')
                ax.plot(1 / bit.S, bit.filter(regex='Q'))  # ['M.Q_A'])
                for ln, nm in zip(ax.lines, bit.filter(regex='Q').columns):
                    ln.set(label=nm[4:])
                ax.xaxis.set_minor_locator(ticker.LogLocator(numticks=10))
                ax.axhline(0, lw=.5, c='k')
                ax.set(xscale='log', xlim=xlim, title=f'{kind} capital density', xlabel='Total loss return period',
                       ylabel='Layer capital rate', ylim=[-0.1, 1.05])
                if ax is ax0:
                    ax.legend()  # loc='lower right')
            # Figure 15.8
            caption = f"(U) {self.case_name}, capital density for {self.case_name}, with {str(self.ports['gross'].dists[self.dgross])} gross and {str(self.ports['net'].dists[self.dnet])} net distortion."
            self._display_plot('U', f, caption)

            # fig 15.11
            self.loss_density_spectrum()

            # fig 15.12, 15.13, 15.14
            logger.info('Figures 15.12, 15.13, 15.14.')
            f, axs = self.get_f_axs(1, 2)
            ax0, ax1 = axs.flat
            # Bodoff allocation...
            lims = [0, self.gross.q(self.reg_p) * 1.1]

            for port, bit, ax in zip([self.gross, self.net], [self.bitg, self.bitn], axs.flat):
                bit.iloc[:, -2:].plot(ax=ax)
                ax.set(xlim=lims, ylim=lims, xlabel='Assets', ylabel='Allocation')
                ax.axvline(port.q(self.reg_p), ls=':', label='Capital')
                ax.legend()
                if ax is ax0:
                    ax.set(title='Gross')
                else:
                    ax.set(title='Net')
            # Figure 15.12
            caption = f'(X) {self.case_name}, percentile layer of capital allocations by asset level, showing {self.reg_p} capital. (Same distortions.)'
            self._display_plot('X', f, caption)

    def make_boundss_p_stars(self, n_tps=1024, n_s=512):
        boundss = {}
        p_stars = {}
        for k, port in self.ports.items():
            bounds = Bounds(port)
            # tvar_cloud -> compute_weights -> tvar_array ; tvar array used below for the graph
            # bounds.tvar_array('total', 1024, a)
            kind = 'interp'
            bounds.tvar_cloud(
                'total', self.pricing_summary.at['P', k], self.pricing_summary.at['a', k], n_tps, n_s, kind)
            boundss[k] = bounds
            p_stars[k] = bounds.p_star('total', self.pricing_summary.at['P', k], self.pricing_summary.at['a',
                                                                                                         k],
                                       kind='tail' if k in ('discrete', 'awkward') else 'interp')
        self.boundss = boundss
        self.p_stars = p_stars
        # return boundss, p_stars

    def make_classic_pricing(self):
        port = self.ports['gross']
        port.calibrate_distortions(ROEs=[self.roe],
                                   Ps=[self.reg_p],
                                   strict='ordered', df=[0, .98])

        # f_roe_fix, put pre-computed approx back
        port.dists['roe'] = self.roe_d
        # and then SET the distortions for net
        self.ports['net'].dists = self.ports['gross'].dists.copy()

        cp = ClassicalPremium(self.ports, self.pricing_summary.loc['P', 'gross'])
        # set power for Fischer
        cp.p = 2

        out = cp.calibrate('gross', 'total', self.pricing_summary.loc['P', 'gross'])
        if self.case_id not in ['cnc', 'tame']:
            if 'Exponential' in out:
                del out['Exponential']
                logger.info('deleting Exponential method')

        # uses unlimited (audit stats mean) losses
        prices = cp.prices('gross', 'total', out)

        bits = {}
        lns = self.ports['gross'].line_names_ex
        bits['gross'] = pd.concat([pd.DataFrame(out.items(), columns=['method', 'parameter']).set_index('method')] +
                                  [pd.DataFrame(cp.prices('gross', ln, out).items(),
                                                columns=['method', 'premium']).set_index('method') for ln in lns]
                                  , axis=1, keys=['parameters'] + lns, names=['line', 'item'])
        bits['net'] = pd.concat([pd.DataFrame(out.items(), columns=['method', 'parameter']).set_index('method')] +
                                [pd.DataFrame(cp.prices('net', ln, out).items(),
                                              columns=['method', 'premium']).set_index('method') for ln in lns]
                                , axis=1, keys=['parameters'] + lns, names=['line', 'item'])
        try:
            bit = pd.concat([bits['gross'], bits['net'].iloc[:, [2, 3]]], keys=bits.keys(),
                            names=['view', 'line', 'item'],
                            axis=1)
        except IndexError:
            logger.warning('Alternative logic...')
            bit = pd.concat([bits['gross'], bits['net']], keys=bits.keys(), names=['view', 'line', 'item'],
                            axis=1)

        bit = bit.droplevel('item', axis=1).swaplevel('view', 'line', axis=1).rename(
            columns={'parameters': ' parameters'}).sort_index(axis=1)

        ll = list(bit.columns)
        ll[0] = (ll[0][0], 'value')

        bit.columns = pd.MultiIndex.from_tuples(ll)

        bit[('total', 'ceded')] = bit[('total', 'gross')] - bit[('total', 'net')]
        bit = bit.sort_values((lns[1], 'gross'))
        emp_means = pd.concat((port.audit_df.T.loc[['EmpMean']] for port in self.ports.values()),
                              keys=self.ports.keys(), names=['view', 'line'], axis=1)
        emp_means[('ceded', 'total')] = emp_means[('gross', 'total')] - emp_means[('net', 'total')]
        emp_means = emp_means.swaplevel(0, 1, axis=1).sort_index(axis=1)
        if self.case_id == 'hs':
            drop_line = 'SCS'
        elif self.case_id == 'tame':
            drop_line = 'A'
        elif self.case_id == 'cnc':
            drop_line = 'NonCat'
        elif self.case_id == 'discrete':
            drop_line = 'X1'
        else:
            drop_line = None
        if drop_line is not None:
            emp_means = emp_means.drop((drop_line, 'net'), axis=1)
        bit = pd.concat((emp_means, bit)).sort_index(ascending=[True, False], axis=1)

        lrs = (bit.loc['EmpMean'] / bit).iloc[:, 1:]

        if self.case_id in ['discrete', 'tame', 'cnc', 'hs'] or 1:
            # not sure why you wouldn't want this?
            no_re_line = self.ports['gross'].line_names.copy()
            no_re_line.remove(self.re_line)
            no_re_line = no_re_line[0]

            sop = bit.copy().drop(columns=[(' parameters', 'value'), ('total', 'ceded')])
            sop[('sop', 'gross')] = sop[(self.re_line, 'gross')] + sop[(no_re_line, 'gross')]
            sop[('sop', 'net')] = sop[(self.re_line, 'net')] + sop[(no_re_line, 'gross')]
            sop[('delta', 'gross')] = sop[('sop', 'gross')] - sop[('total', 'gross')]
            sop[('delta', 'net')] = sop[('sop', 'net')] - sop[('total', 'net')]
            sop = sop[['total', 'sop', 'delta']]
            sop = sop.sort_index(axis=1, ascending=[False, True])
            sop.index.name = 'method'
        else:
            sop = None

        bit.index.name = 'method'
        lrs.index.name = 'method'

        # self.port = port
        # self.cp = cp
        self.prices = prices
        # self.bits = bits
        self.lns = lns
        self.classic_pricing = bit
        self.lrs = lrs
        # self.no_re_line = no_re_line
        self.sop = sop

        # Table 13.1
        tab13_1 = pd.DataFrame(index=self.gross.line_names_ex)
        tab13_1.index.name = 'Unit'
        vd = self.gross.var_dict(self.reg_p)
        tab13_1['a'] = vd.values()
        tab13_1['E[Xi(a)]'] = \
            self.gross.density_df.filter(regex='exa_[A-Zt]').rename(columns=lambda x: x.replace('exa_', '')).loc[
                vd['total']]
        tab13_1['E[Xi ∧ ai]'] = [self.gross.density_df.loc[v, f'lev_{k}'] for k, v in vd.items()]
        tab13_1 = tab13_1.rename(index={'total': 'Total'})
        tab13_1.loc['SoP'] = tab13_1.iloc[:-1].sum()
        self.tab13_1 = tab13_1

    def make_modern_monoline(self):
        bitsgross = []
        for port in self.ports.values():
            # unlike
            bitgross = stand_alone_pricing(port, self.roe_d, p=self.reg_p, kind='var')
            bitgross = bitgross.swaplevel(0, 1).sort_index(ascending=[1, 0]).rename(
                index={'traditional - no default': 'no default', f'sa {self.roe_d}': 'constant roe'})
            # if len(bitsgross) == 0:
            #     display(bitgross)
            bitsgross.append(bitgross)
        bit = pd.concat([bg.loc[['L', 'M', 'P', 'LR', 'Q', 'ROE', 'PQ', 'a']].drop(
            'constant roe', axis=0, level=1)
            for bg in bitsgross],
            axis=1, keys=self.ports.keys(), names=['portfolio', 'line'])

        rob = bit.rename(index=universal_renamer).rename(index={'Traditional': 'With Default'}) \
            .swaplevel(axis=0).sort_index()
        order = np.array([3, 5, 6, 4, 1, 7, 2, 0])
        order = np.hstack((order, order + 8))
        rob = rob.iloc[order, [0, 1, 2, 3, 4, 6, 7]]
        rob.index.names = ['method', 'statistic']
        self.modern_monoline_sa = rob

        # add self.blend_d to list of distortions
        for port in self.ports.values():
            # add mass_hints...
            if 'roe' not in port.dists:
                port.dists['roe'] = self.dist_dict['roe']
            if 'blend' not in port.dists and 'blend' in self.dist_dict:
                port.dists['blend'] = self.dist_dict['blend']
        # Ch. Modern Mono Practice across distortions
        # all using GROSS calibrated distortions:
        distortions = [self.ports['gross'].dists[dn] for dn in ['roe', 'ph', 'wang', 'dual', 'tvar', 'blend']]
        bit_apply_gross = []
        for port in self.ports.values():
            temp = stand_alone_pricing(port, distortions, p=self.reg_p, kind='var')
            temp = temp.swaplevel(0, 1).sort_index(ascending=[1, 0])
            bit_apply_gross.append(temp.filter(regex='sa', axis=0).rename(index=distortion_namer))
        bit = pd.concat([bg.loc[['L', 'M', 'P', 'LR', 'Q', 'ROE', 'PQ', 'a']]
                         for bg in bit_apply_gross],
                        axis=1, keys=self.ports.keys(), names=['portfolio', 'line'])

        rob2 = bit.rename(index=universal_renamer)  # .swaplevel(axis=0)
        # order = np.array([5, 3, 1, 4, 2, 0])
        order = np.array([0, 4, 2, 5, 3, 1])
        order = np.hstack([0] + [order + i * 6 for i in range(1, 7)] + [-6])
        if self.case_id == 'tame':
            cols = [0, 1, 2, 3, 5, 6, 7]
        else:
            cols = [0, 1, 2, 3, 4, 6, 7]
        rob2 = rob2.iloc[order, cols]
        rob2.index.names = ['statistic', 'distortion']
        # is this different?
        self.modern_monoline_by_distortion = rob2

    def make_ad_comps(self):
        # ad = analyze distortion: combines all pricing methods into one summary data frame 
        # TIME CONSUMING STEP...computed here for all allocations used here and Ch 090 110 for the HUSCS line
        # remember above, we replaced the dists for net with that calibrated on gross
        # print(self.ports['gross'].dists == self.ports['net'].dists)
        ad_compss = {}
        for k, port in self.ports.items():
            if k not in ad_compss:
                ad_comps = port.analyze_distortions(p=self.reg_p,
                                                    efficient=False,
                                                    regex='ph|dual|wang|tvar|roe|blend')
                ad_compss[k] = ad_comps
                if k == 'net':
                    # for classical CCoC need to just use roe distortion
                    ad_compss['net_classical'] = port.analyze_distortions(p=self.reg_p,
                                                                          efficient=False,
                                                                          regex='roe')
        summaries = {}
        for k, ad_comps in ad_compss.items():
            deets = ad_comps.comp_df.swaplevel(0, 1).sort_index(ascending=(1, 0))
            summary = pd.concat((
                deets.loc[[('L', 'EL')]],
                deets.loc[CaseStudy._stats_[1:]]))
            summaries[k] = summary
        self.ad_compss = ad_compss
        self.summaries = summaries
        # this is a bit mix n match
        self.classical_pricing = extract_sort_order(self.summaries, self._classical_, True)
        self.distortion_pricing = extract_sort_order(self.summaries, self._dist_, False)

    def make_bodoff_comparison(self):
        # tables 15.38-40 and Fig15.12-14
        # note the allocation is assets, not capital
        bitg = self.gross.augmented_df.filter(regex='loss|exi_xgta_[A-Z]|F|^S$')
        bitn = self.net.augmented_df.filter(regex='loss|exi_xgta_[A-Z]|F|^S$')
        for bit in [bitg, bitn]:
            bit.set_index('loss', inplace=True)
            for ln in self.gross.line_names:
                bit[ln] = bit[f'exi_xgta_{ln}'].shift(1, fill_value=0).cumsum() * self.gross.bs

        cbit = self.distortion_pricing.loc['a'].copy()
        r = f'^{self.gross.line_names[0]}|^{self.gross.line_names[1]}'
        g = bitg.loc[self.gross.q(self.reg_p), :].filter(regex=r)
        n = bitn.loc[self.net.q(self.reg_p), :].filter(regex=r)
        g['total'] = g.sum()
        n['total'] = n.sum()
        cbit.loc['PLC', :] = 0

        for i in g.index:
            cbit.loc['PLC', ('gross', i)] = g[i]
            cbit.loc['PLC', ('net', i)] = n[i]
        cbit.loc['PLC', ('ceded', 'diff')] = cbit.loc['PLC', ('gross', 'total')] - cbit.loc['PLC', ('net', 'total')]
        # TODO rename these!
        self.bitg = bitg
        self.bitn = bitn
        self.bodoff_compare = cbit

    def make_progression(self):
        """
        Integrate modern stand-alone and portfolio pricing

        :return:
        """
        vas = ['Premium', 'Loss Ratio', 'Rate of Return']
        dns = ['CCoC', 'PH', 'Wang', 'Dual', 'TVaR', 'Blend']
        idx = product(vas, dns)
        bit_sa = self.modern_monoline_by_distortion.loc[idx]

        vas = ['P', 'LR', 'ROE']
        dns = ['Dist roe', 'Dist ph', 'Dist wang', 'Dist dual', 'Dist tvar', 'Dist blend']
        idx = product(vas, dns)
        renamer = {'P': 'Premium', 'LR': 'Loss Rato', 'ROE': 'Rate of return', 'Dist roe': 'CCoC', 'Dist ph': 'PH',
                   'Dist wang': 'Wang', 'Dist dual': 'Dual', 'Dist tvar': 'TVaR', 'Dist blend': 'Blend'}
        bit_d = self.distortion_pricing.loc[idx]
        bit_d = bit_d.drop(('ceded', 'diff'), axis=1).rename(index=renamer)
        bit = pd.concat((bit_sa, bit_d), keys=['Stand-alone', 'Allocated'])
        bit.index.names = ['Approach', 'Statistic', 'Distortion']
        bit.columns.names = ['Perspective', 'Unit']

        # and subset to just the premium walk
        b0 = urn(bit.xs('Premium', axis=0, level=1)['gross'].fillna('')).copy()
        b0.loc['Allocated', 'SoP'] = b0.loc['Allocated', 'Total'].values
        b0 = b0.drop(columns=['Total'])
        b0.columns = list(b0.columns[:-1]) + ['Total']
        # difference
        # TODO SORT OUT - why are there missing values?
        b0d = b0.loc['Stand-Alone'].replace('', 0.) - b0.loc['Allocated']
        walk = pd.concat((b0d,
                          b0d / b0.loc['Stand-Alone'].replace('', 1),
                          b0d.div(b0d['Total'].values, axis=0)
                          ),
                         keys=['Pooling ¤ Benefit', 'Pct Premium Reduction', 'Pct of Benefit'],
                         axis=1,
                         names=['View', 'Unit']
                         )
        self.progression = bit
        self.walk = walk

    def make_all(self, dnet='', dgross='', n_tps=256, n_s=128):
        """
        run code to make all the graphs and exhibits
        applies book default distortions or user selected

        Was n_tps=1024, n_s=512 which seems overkill
        """
        # self.make_audit_exhibits()
        # print('audit exhibits done')
        self.make_boundss_p_stars(n_tps, n_s)
        process_memory()
        logger.log(35, 'boundss and pstar done')
        self.make_classic_pricing()
        process_memory()
        logger.log(35, 'classic pricing done')
        self.make_modern_monoline()
        process_memory()
        logger.log(35, 'modern monoline done')
        self.make_ad_comps()
        process_memory()
        logger.log(35, 'ad comps done')
        self.apply_distortions(dnet, dgross)
        process_memory()
        logger.log(35, 'apply distortions done')
        self.make_bodoff_comparison()
        process_memory()
        logger.log(35, 'Bodoff exhibits done')

    @staticmethod
    def extract_image(p, ref, w=100, caption=''):
        import base64
        if caption == '':
            caption = 'some made up bullshit'
        data_uri = base64.b64encode(p.read_bytes()).decode('utf-8')
        img_tag = f'''
<figure>
<img src="data:image/png;base64,{data_uri}" alt="Figure {ref}" style="width:{w}%">
<figcaption>{caption}</figcaption>
</figure>
'''
        return img_tag


def extract_sort_order(summaries, _varlist_, classical=False):
    '''
    Pull out exhibits. Note difference: classical uses net_classical, calibrated to roe
    non-classical uses pricing calibrated to
    '''
    parts = ['gross']
    if classical:
        parts.append('net_classical')
    else:
        parts.append('net')
    bit = pd.concat([summaries[k] for k in parts], keys=['gross', 'net'], axis=1)
    ceded_name = 'diff'
    bit[('ceded', ceded_name)] = bit[('gross', 'total')] - bit[('net', 'total')]
    bit.loc['LR', ('ceded', ceded_name)] = bit.loc['L', ('ceded', ceded_name)].to_numpy() / bit.loc[
        'P', ('ceded', ceded_name)].to_numpy()
    bit.loc['ROE', ('ceded', ceded_name)] = bit.loc['M', ('ceded', ceded_name)].to_numpy() / bit.loc[
        'Q', ('ceded', ceded_name)].to_numpy()
    bit.loc['PQ', ('ceded', ceded_name)] = bit.loc['P', ('ceded', ceded_name)].to_numpy() / bit.loc[
        'Q', ('ceded', ceded_name)].to_numpy()
    df = bit.loc[(slice(None), _varlist_), :].copy()
    df['dm1'] = [CaseStudy._stats_.index(i) for i in df.index.get_level_values(0)]
    df['dm2'] = [_varlist_.index(i) for i in df.index.get_level_values(1)]
    df = df.sort_values(['dm1', 'dm2'])
    df = df.drop(columns=['dm1', 'dm2'])
    return df


def g_ins_stats(axi, dn0, dist, ls='-'):
    """
    Six part plot with EL, premium, loss ratio, profit, layer ROE and P:S for g
    axi = axis iterator
    """
    g = dist.g
    N = 1000
    ps = np.linspace(1 / (2 * N), 1, N, endpoint=False)
    gs = g(ps)

    # ['Premium, $g(s)$', 'Loss Ratio, $s/g(s)$', 'Markup, $g(s)/s$',
    #  'Margin, $g(s)-s$', 'Discount Rate, $(g(s)-s)/(1-s)$', 'Premium Leverage, $g(s)/(1-g(s))$']):

    dn = {'ph': "PH",
          'wang': "Wang",
          'dual': "Dual",
          'roe': 'CCoC',
          'tvar': 'TVaR',
          'blend': 'Blend'}[dn0]
    if dn0 == 'blend':
        dn = 'Blend'
    elif dn0 == 'roe':
        # TODO HORRIBLE
        dn = f'{dn}, 0.10'
    else:
        dn = f'{dn}, {dist.shape:.3f}'

    # FIG: six panel display
    for i, ys, key in zip(range(1, 7),
                          [gs, ps / gs, gs / ps, gs - ps, (gs - ps) / (1 - ps), gs / (1 - gs)],
                          ['Premium', 'Loss ratio', 'Markup',
                           'Margin', 'Discount rate', 'Premium leverage']):
        ax = next(axi)
        if i == 1:
            ax.plot(ps, ys, ls=ls, label=dn)
        else:
            ax.plot(ps, ys, ls=ls)
        if i in [1, 2, 4]:
            ax.plot(ps, ps, color='C2', linewidth=0.25)
        ax.set(title=key)
        if i == 3 or i == 6:
            ax.axis([-0.025, 1.025, 0, 5])
            ax.set(aspect=1 / 5)
        else:
            ax.axis([-0.025, 1.025, -0.025, 1.025])
            ax.set(aspect=1)
        if i == 1:
            ax.legend(loc='lower right')


# ============================================================================
# ============================================================================
# ============================================================================
# Similar Risks IME Paper Functions
def show3d(df, height=600, xlabel="p_upper", ylabel="p_lower", zlabel="Price", initialCamera=None):
    """
    quick plot of a Bounds cloud_df view FRAGILE

    def plot3D(X, Y, Z, height=600, xlabel="X", ylabel="Y", zlabel="Z", initialCamera=None):

        Javascript 3d graphics.
        Sample call

        X, Y = np.meshgrid(np.linspace(-3,3,50),np.linspace(-3,3,50))
        Z = np.sin(X**2 + Y**2)**2/(X**2+Y**2)
        plot3D(X, Y, Z)

        :param X:
        :param Y:
        :param Z:
        :param height:
        :param xlabel:
        :param ylabel:
        :param zlabel:
        :param initialCamera:
        :return:

    """
    bit = df.t1.unstack()
    X, Y = np.meshgrid(bit.columns, bit.index)
    Z = bit.to_numpy()

    options = {
        "width": "100%",
        "style": "surface",
        "showPerspective": True,
        "showGrid": True,
        "showShadow": False,
        "keepAspectRatio": True,
        "height": str(height) + "px",
        "xlabel": xlabel
    }

    if initialCamera:
        options["cameraPosition"] = initialCamera

    data = [{"x": X[y, x], "y": Y[y, x], "z": Z[y, x]} for y in range(X.shape[0]) for x in range(X.shape[1])]
    visCode = r"""
       <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" type="text/css" rel="stylesheet" />
       <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
       <div id="pos" style="top:0px;left:0px;position:absolute;"></div>
       <div id="visualization"></div>
       <script type="text/javascript">
        var data = new vis.DataSet();
        data.add(""" + json.dumps(data) + """);
        var options = """ + json.dumps(options) + """;
        var container = document.getElementById("visualization");
        var graph3d = new vis.Graph3d(container, data, options);
        graph3d.on("cameraPositionChange", function(evt)
        {
            elem = document.getElementById("pos");
            elem.innerHTML = "H: " + evt.horizontal + "<br>V: " + evt.vertical + "<br>D: " + evt.distance;
        });
       </script>
    """
    htmlCode = "<iframe srcdoc='" + visCode + "' width='100%' height='" + str(
        height) + "px' style='border:0;' scrolling='no'> </iframe>"
    display(HTML(htmlCode))


def plot_max_min(self, ax):
    """
    Extracted from bounds, self=Bounds object
    """
    ax.fill_between(self.cloud_df.index, self.cloud_df.min(1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
    self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=0.5, ls='-', c='w')
    self.cloud_df.max(1).plot(ax=ax, label="_nolegend_", lw=0.5, ls='-', c='w')


def plot_lee(port, ax, c, lw=2):
    """
    Lee diagram by hand
    """
    p_ = np.linspace(0, 1, 1001)
    qs = [port.q(p) for p in p_]
    ax.step(p_, qs, lw=lw, c=c)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, max(qs) + .05], title=f'Lee Diagram {port.name}', aspect='equal')


class ClassicalPremium(object):
    """
    manage classical premium examples

        Net, no loading
        Expected value, constant loading
        Maximum loss --> no
        VaR (as proxy for maximum loss)
        Variance
        Std Dev
        Semi-variance (Artzner p. 210)
        Exponential (zero utility, convex!)
        Esscher

    Originally in hack.py

    """

    __classical_methods__ = ['Expected Value', 'VaR', 'Variance', 'Standard Deviation',
                             'Semi-Variance', 'Exponential', 'Esscher', 'Dutch', 'Fischer']

    def __init__(self, ports, calibration_premium):
        """
        set up and calibrate pricing methods
        """
        self.ports = ports
        self.calibration_premium = calibration_premium

    def distribution(self, port_name, line_name):
        """
        classical methods all depend on the distribution...so pull it out
        pull the object that will provide q etc.
        pull the audit stats

        """

        df = self.ports[port_name].density_df.filter(regex=f'p_{line_name}|loss'). \
            rename(columns={f'p_{line_name}': 'p'})
        df['F'] = df.p.cumsum()
        if line_name == 'total':
            ob = self.ports[port_name]
        else:
            ob = self.ports[port_name][line_name]
        stats = self.ports[port_name].audit_df.T[line_name]
        mn = stats['EmpEX1']
        var = stats['EmpEX2'] - stats['EmpEX1'] ** 2
        sd = var ** 0.5
        return df, ob, stats, mn, var, sd

    def calibrate(self, port_name, line_name, calibration_premium, df=None, ob=None, stats=None, mn=None, var=None,
                  sd=None):
        """
        calibrate all methods...
        """
        from scipy.optimize import newton
        self.calibration_premium = calibration_premium

        # gather inputs
        if df is None:
            df, ob, stats, mn, var, sd = self.distribution(port_name, line_name)

        ans = {}
        ans['Expected Value'] = calibration_premium / mn - 1
        ans['VaR'] = float(ob.cdf(calibration_premium))
        ans['Variance'] = (calibration_premium - mn) / var
        ans['Standard Deviation'] = (calibration_premium - mn) / sd

        for method in self.__classical_methods__:
            if method not in ans:
                if method in ['Exponential', 'Esscher']:
                    x0 = 1e-10
                elif method == 'TVaR':
                    x0 = ans['VaR'] - 0.025
                else:
                    x0 = 0.5
                try:
                    a = newton(lambda x: self.price(x, port_name, line_name, method, df, ob, stats, mn, var,
                                                    sd) - calibration_premium, x0=x0)
                    ans[method] = float(a)
                except RuntimeError as e:
                    print(method, e)
        return ans

    def prices(self, port_name, line_name, method_dict):
        """
        run lots of prices

        """
        df, ob, stats, mn, var, sd = self.distribution(port_name, line_name)
        ans = {}
        for method, param in method_dict.items():
            ans[method] = self.price(param, port_name, line_name, method, df, ob, stats, mn, var, sd)
        return ans

    def price(self, param, port_name, line_name, method, df=None, ob=None, stats=None, mn=None, var=None, sd=None):
        """
        apply method to port_name, line_name with parameter(s) (all one param)
        these are all classical methods

        method_dict = {method name : param }
        param = float | [float param, p ge 1 value] latter fro Fischer method
        """
        if df is None:
            df, ob, stats, mn, var, sd = self.distribution(port_name, line_name)

        if method == 'Expected Value':
            return mn * (1 + param)
        if method == 'VaR':
            return ob.q(param)
        if method == 'TVaR':
            return ob.tvar(param)
        if method == 'Variance':
            return mn + param * var
        if method == 'Standard Deviation':
            return mn + param * sd
        if method == 'Semi-Variance':
            sd_plus = np.sum(np.maximum(0, df.loss - mn) ** 2 * df.p)
            return mn + param * sd_plus
        if method == 'Exponential':
            # (1/k) ln E[e^kX]
            eax = np.sum(np.exp(param * df.loss) * df.p)
            return (1 / param) * np.log(eax)
        if method == 'Esscher':
            # E[Xe^aX] / E[e^aX]
            eax = np.sum(np.exp(param * df.loss) * df.p)
            exax = np.sum(df.loss * np.exp(param * df.loss) * df.p)
            return exax / eax
        if method == 'Dutch':
            excess = np.sum(np.maximum(df.loss - mn, 0) * df.p)
            return mn + param * excess
        if method == 'Fischer':
            # remember you must set p!
            excess = np.sum(np.maximum(df.loss - mn, 0) ** self.p * df.p) ** (1 / self.p)
            return mn + param * excess

    def pricing_exhibit(self, port_name, line_name, calibration_premium, re_line):
        """
        calibrate and apply to all other portfolios

        """

        df, ob, stats, mn, var, sd = self.distribution(port_name, line_name)
        parameters = self.calibrate(port_name, line_name, calibration_premium, df, ob, stats, mn, var, sd)

        df = pd.DataFrame({(pfn, ln): pd.Series(self.prices(pfn, ln, parameters))
                           for pfn in self.ports.keys()
                           for ln in self.ports['gross'].line_names_ex})
        df[('ceded', re_line)] = df[('gross', re_line)] - df[('net', re_line)]

        el = pd.Series(np.hstack((self.ports['gross'].audit_df['EmpMean'].values,
                                  self.ports['net'].audit_df['EmpMean'].values,
                                  mn - self.ports['net'].audit_df.loc[re_line, 'EmpMean'])),
                       index=df.columns)
        df = pd.concat((df, el / df.replace(0, np.nan)), keys=['P', 'LR'], names=['stat', 'view', 'line'], axis=1)
        return df

    def illustrate(self, port_name, line_name, ax, margin,
                   *, p=0, K=0, n_big=10000, n_sample=25, show_bounds=True,
                   padding=2):
        '''
        illustrate simulations at p level probability
        probability level determines capital or capital K input
        margin: premium = (1 + margin) * EL, margin = rho
        n_big = number of policies - max of horizontal axis
        n_sample = number of iterations to plot

        Theoretic bounds use the actual moments, simulated use those
        from the process being estimated

        From common_scripts Pentagon
        took out re-computation...

        '''
        if line_name == 'total':
            print('Error - cannot use total...')
            raise ValueError()
            # ag = self.ports[port_name]
        else:
            ag = self.ports[port_name][line_name]

        if p and K == 0:
            self.ruin, self.u, self.mean, self._dfi = ag.cramer_lundberg(margin, kind='interpolate', padding=padding)
            K = self.u(p)
            # print(f'Using K={K}')
        elif K == 0:
            raise ValueError('Must input one of K and p')

        ea = np.sum(ag.density_df.loss * ag.density_df.p)
        ans = [(-1, ea)]
        ns = np.arange(n_big)
        prems = K + (1 + margin) * ea * ns
        means = K + margin * ea * ns
        # transparency of lines...
        alpha = min(1, 50 / n_sample)
        bombs = 0
        for i in range(n_sample):
            samp = ag.density_df.loss.sample(
                n=n_big, weights=ag.density_df.p, replace=True)
            ans.append((i, np.mean(samp), np.max(samp)))
            ser = prems - samp.cumsum()
            idzero = np.argmax(ser <= 0)
            if idzero > 0:
                ser.iloc[idzero:] = ser.iloc[idzero]
            rc = np.random.rand()
            ax.plot(ns, ser, ls='-', lw=0.5,
                    color=(rc, rc, rc), alpha=alpha)
            if idzero:
                ax.plot(ns[idzero], ser.iloc[idzero], 'x', ms=5)
                bombs += 1
        ax.plot(ns, means, 'k-', linewidth=2)
        ef = EngFormatter(3, True)
        ax.set(xlabel='Number of risks', ylabel=f'Cumulative capital, $u_0={ef(K)}$')
        # LIL bounds
        lln = np.nan_to_num(np.log(abs(np.log(ns))))
        lln[lln < 0] = 0

        if show_bounds:
            var_a = (ag.agg_m * ag.agg_cv) ** 2
            lb = means - np.sqrt(2.0 * var_a * ns * lln)
            ub = means + np.sqrt(2.0 * var_a * ns * lln)
            ax.plot(ns, lb, '--', lw=1)
            ax.plot(ns, ub, '--', lw=1)
        # titles etc.
        # title = f'{ag.name:s} at {margin:.3f} margin\nFailure rate: {bombs} / {n_sample} = {bombs / n_sample:.3f}'
        title = f'{bombs}/{n_sample} = {bombs / n_sample:.3f} ruins'
        # if p:
        #     title += f' (p={p:.3f})'
        ax.set(title=title)
        # make xy plot
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))

        return pd.DataFrame(ans, columns=['n', 'ea', 'min'])


# code from common_scripts
#  `enhance_portfolio`
#  `TensePortfolio`
#  `RelaxedPortfolio`
#  `pricing`
#  `twelve_plot`
#  `bivariate_density_plots`
#  `ch04_macro_market_stats` --> macro_market_graphs
#  `Bounds` class
#  `stand_alone_pricing`


def macro_market_graphs(axi, port, dn, rp):
    """
    Create more succinct 4x4 graph to illustrate macro market structure and result, LR, P:S and ROE, etc.

    Use a port object with calibrated distortion

    see ch04_macro_market_stats_original

    from: common_scripts.py ch04_macro_market_stats
    updated line colors

    June 2022: removed all color='k'

    :param dist:
    :param rp: return period
    :param sigma:
    :return:
    """

    dist = port.dists[dn]
    g = dist.g
    dn = {'ph': "PH",
          'wang': "Wang",
          'dual': "Dual",
          'roe': 'CCoC',
          'tvar': 'TVaR',
          'blend': 'Blend'}[dn]

    # figure assets and basic stats
    A = port.q(1 - 1 / rp)

    xs = port.density_df.loss.to_numpy()
    sxs = port.density_df.S.to_numpy()
    gsxs = g(sxs)

    h = xs[1]
    el = cumtrapz(sxs, dx=h, initial=0)
    prem = cumtrapz(gsxs, dx=h, initial=0)

    # marginal layer statistics
    # be careful about divide by zero
    eps = np.finfo(float).eps
    iota = np.where(gsxs > 1 - eps, np.inf, (gsxs - sxs) / (1 - gsxs))
    nu = np.where(np.isinf(iota), np.nan, 1 / (1 + iota))
    delta = 1 - nu
    # delta = 1 - nu
    lr = sxs / gsxs
    leverage = gsxs / (1 - gsxs)

    # cumulative statistics, from the bottom to the limit
    eps = A * eps * 100
    ciota = np.where(prem > xs - eps, np.inf, (prem - el) / (xs - prem))
    cnu = np.where(np.isinf(ciota), np.nan, 1 / (1 + ciota))
    clr = np.where(prem == 0, np.nan, el / prem)
    cdelta = 1 - cnu

    a = next(axi)
    a.plot(xs, prem, '-', label='Prem')
    a.plot(xs, el, '--', label='Loss')

    a.set(title=dn)  # f'$\\bar P$ and $\\bar S$, {dn}')
    a.legend(loc='lower right')

    a = next(axi)
    a.plot(xs, prem, '-', label='premium')
    a.plot(xs, xs, '-', lw=.75, label='assets')
    a.plot(xs, xs - prem, '-.', label='capital')
    a.set(title='$a, \\bar P$ and $\\bar Q$')
    a.legend(loc='upper left')

    a = next(axi)
    a.plot(xs, clr, '-', label='cumul')
    a.plot(xs, lr, ':', label='layer')
    a.set(ylim=[-0.01, 1.01], title='Loss ratio')
    a.legend(loc='lower right')

    a = next(axi)
    a.plot(xs, cdelta, '-', label='cumul')
    a.plot(xs, delta, ':', label='layer')
    a.set(ylim=[-0.01, 1.01], title='Discount rate $\\delta$')
    a.legend(loc='upper right')


# def RelaxedPortfolio(port_or_prog, log2=13, bs=0, padding=2, approx_freq_ge=100):
#     """
#     enable basic set of exhibits
#     checks to see if the enhanced methods are in the class and adds to the instance if not
#
#     executes a recalc with sensible defaults
#
#     see enhance_portfolio for a list of added methods
#     from common_scripts.py
#
#     """
#     if type(port_or_prog) == str:
#         # program
#         uw = agg.Underwriter(create_all=False)
#         # new style return
#         out = uw(port_or_prog)
#         # TODO: sort out
#         print(out, 'FIGURE OUT WHAT TO DO'*4)
#         raise ValueError('asdf')
#     else:
#         port = port_or_prog
#
#     if port.density_df is None:
#         # not recomputed
#         if bs == 0:
#             bs = port.best_bucket(log2)
#         logging.warning(
#             f'Recomputing {port.name} with log2={log2}, bs={bs if bs >= 1 else 1/bs if bs >= 1 else int(1 / bs)}...')
#         port.update(log2=log2, bs=bs, padding=padding, remove_fuzz=True, add_exa=True,
#                     approx_freq_ge=approx_freq_ge, trim_df=False)
#
#     enhance_portfolio(port)
#     # create standard exhibit
#     # port.basic_loss_statistics()
#     return port
#
#
# def TensePortfolio(port_or_prog, log2=13, bs=0, padding=2, approx_freq_ge=100,
#                    dist_name='wang', a=0, p=0, ROE=0.10, r0=0.02, df=[0.0, 0.9], mass_hints=None,
#                    dist=None):
#     """
#     relaxed plus more...where you need a distortion or generally more choices are required
#
#     dist_name for analysis
#
#     dist = just pass in a distortion directly...it will be applied.
#
#     also handles populating the reserve example templates
#
#     see enhance_portfolio for a list of added methods
#
#     from common_scripts.py
#
#     """
#
#     port = RelaxedPortfolio(port_or_prog, log2, bs, padding, approx_freq_ge=approx_freq_ge)
#     a, p = port.set_a_p(a, p)
#     port.a = a
#     port.p = p
#     port.ROE = ROE
#
#     if dist is None:
#         recalibrate = False
#         if port.dist_ans is None:
#             recalibrate = True
#         if port.distortion is None:
#             port.dist_name = dist_name
#             recalibrate = True
#         else:
#             # have both...make sure consistent
#             port.dist_name = dist_name
#             if dist_name != port.distortion.name:
#                 # need to change distortion and recalibrate
#                 recalibrate = True
#
#         if recalibrate:
#             logging.warning(f'Calibrating distortions for {port.name} with ROE={ROE}, a={a}, p={p}...')
#             port.calibrate_distortions(r0=r0, df=df, ROEs=[ROE], As=[a], strict=False)
#
#             port.apply_distortion(port.dists[dist_name], plots=None, df_in=None, create_augmented=True,
#                                   mass_hints=mass_hints)
#             port.ad_ans = port.analyze_distortion(dist_name, p=p, A=a, ROE=ROE, use_self=True, plot=False,
#                                                   mass_hints=mass_hints)
#
#             # one line premium and capital summary from dist calibration
#             # port.distortion_information()
#             # port.distortion_calibration()
#     else:
#         # handed in a distortion
#         logging.warning(f'Applying input distortion for {port.name} with ROE={ROE}, a={a}, p={p}...')
#         port.apply_distortion(dist, plots=None, df_in=None, create_augmented=True, mass_hints=mass_hints)
#         port.ad_ans = port.analyze_distortion(dist, use_self=True, plot=False, mass_hints=mass_hints)
#
#         # one line premium and capital summary from dist calibration
#         # port.distortion_information()
#         # port.distortion_calibration()
#
#     # levels for the distortion calibration
#     # dm = port.dist_ans.xs(port.dist_name, level=2, axis=0).reset_index()
#     # dma = float(dm['$a$'])
#     # dmp = port.cdf(port.a)
#     # dmROE = float(dm['ROE'])
#     # logging.warning(f' Final calibration returned a={dma}, p={dmp}, ROE={dmROE}')
#
#     # make the standard exhibits
#     # port.distortion_information()
#     # port.distortion_calibration()
#     return port


def enhance_portfolio(port, force=False):
    """
    Add all the enhanced exhibits methods to port.

    Methods defined within this function.

    From common_scripts.py
    June 2022 took out options that needed a jinja template (reserve story etc.)

    Added Methods

    Exhibit creators (EX_name)
    --------------------------
        1. basic_loss_statistics
        2. distortion_information
        3. distortion_calibration
        4. premium_capital
        5. multi_premium_capital
        6. accounting_economic_balance_sheet
            compares best estimate, default adjusted, risk adjusted values

        Exhibits 7-9 are for reserving
        DROPPED 7. margin_earned (by year)
        DROPPED 8. year_end_option_analysis (implied stand alone vs pooled analysis)

        Run a distortion and compare allocations
        9. compare_allocations
            creates:
                EX_natural_allocation_summary
                EX_allocated_capital_comparison
                EX_margin_comparison
                EX_return_on_allocated_capital_comparison

    Exhibit Utilities
    -----------------
        10. make_all
            runs all of 1-9 with sensible defaults

        11. show
            shows all exhibits, with exhibit title
            uses `self.dir` to find all attributes EX_

        12. qi
            quick info: the basic_loss_stats plus a density plot

    Graphics
    --------
        DROPPED 13. density_plot
        14. profit_segment_plot
            plots given lines S, gS and shades the profit segment between the two
            lines plotted on a stand-alone basis; optional transs allows shifting up/down
        15. natural_profit_segment_plot
            plot kappa = EX_i|X against F and gF
            compares the natural allocation with stand alone pricing
        DROPPED 16. alpha_beta_four_plot
            alpha, beta; layer and cumulative margin plots
        DROPPED 17. alpha_beta_four_plot2 (for two line portfolios )
            lee and not lee orientations (lee orientation hard to parse)
            S, aS; gS, b gS separately by line
            S, aS, gS, bGS  for each line [these are most useful plots]
        18. biv_contour_plot
            bivariate plot of marginals with some x+y=constant lines
        DROPPED 19. reserve_story_md

    Reserve Template Populators
    ---------------------------
        DROPPED 20. reserve_runoff_md
        DROPPED 21. reserve_two_step_md
        22. nice_program

    Other
    -----
        DROPPED 23. show_md
        DROPPED 24. report_args
        DROPPED 25. save
        26. density_sample: stratified sample from density_df


    Sample Runner
    =============
    ```python

        from common_header import *
        get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
        import common_scripts as cs

        port = cs.TensePortfolio('''
        port CAS
            agg Thick 5000 loss 100 x 0 sev lognorm 10 cv 20 mixed sig 0.35 0.6
            agg Thin 5000 loss 100 x 0 sev lognorm 10 cv 20 poisson
        ''', dist_name='wang', a=20000, ROE=0.1, log2=16, bs=1, padding=2)

        # port.make_all() will update all exhibits with sensible defaults

        port.basic_loss_statistics(p=.9999)
        display(port.EX_basic_loss_statistics)

        port.distortion_information()
        display(port.EX_distortion_information)

        port.distortion_calibration()
        display(port.EX_distortion_calibration)

        port.premium_capital(a=20000)
        display(port.EX_premium_capital)

        port.multi_premium_capital(As=[15000, 20000, 25000])
        display(port.EX_multi_premium_capital)

        port.accounting_economic_balance_sheet(a=20000)
        display(port.EX_accounting_economic_balance_sheets)

        port.margin_earned(a=20000)
        display(port.EX_margin_earned)

        port.year_end_option_analysis('Thin', a=20000)
        display(port.EX_year_end_option_analysis)

        port.compare_allocations('wang', ROE=0.1, a=20000)
        display(port.EX_natural_allocation_summary)
        display(port.EX_allocated_capital_comparison)
        display(port.EX_margin_comparison)
        display(port.EX_return_on_allocated_capital_comparison)

        port.show()

        port.qi()

        f, axs = smfig(1,2, (7,3))
        a1, a2 = axs.flat
        port.density_plot(f, a1, a2, p=0.999999)

        smfig = grt.FigureManager(cycle='c', color_mode='c', legend_font='medium')
        f, a = smfig(1,1,(4,6))
        port.profit_segment_plot(a, 0.999, ['total', 'Thick', 'Thin'],
                                     [2,0,1,0], [0,0,0], 'ph')

        f, a = smfig(1,1,(6,10))
        port.natural_profit_segment_plot(a, 0.999, ['total', 'Thick', 'Thin'],
                                     [2,0,1,0], [0,0,0])
        port.profit_segment_plot(a, 0.999, ['Thick', 'Thin'],
                                     [3,4], [0,0], 'wang')
        a.legend()

        f, axs = smfig(2,2,(8,6))
        port.alpha_beta_four_plot(axs, 20000)

        f, axs = smfig(2,2,(8,6))
        port.alpha_beta_four_plot2(axs, 20000, 20000, 'xlee')

        f, axs = smfig(2,2,(8,6))
        port.alpha_beta_four_plot2(axs, 20000, 20000, 'lee')

        aug_df = port.augmented_df
        f, axs = smfig(1,2, (10,5), sharey=True)
        a1, a2 = axs.flat
        bigx = 20000
        bit = aug_df.loc[0:, :].filter(regex='exeqa_(T|t)').copy()
        bit.loc[bit.exeqa_Thick==0, ['exeqa_Thick', 'exeqa_Thin']] = np.nan
        bit.rename(columns=port.renamer).sort_index(1).plot(ax=a1)
        a1.set(xlim=[0,bigx], ylim=[0,bigx], xlabel='Total Loss', ylabel="Conditional Line Loss");
        a1.set(aspect='equal', title='Conditional Expectations\nBy Line')
        port.biv_contour_plot(f, a2, 5, bigx, 100, log=False, cmap='viridis_r', min_density=1e-12)

    ```
    force = do even if exist...force an update
    """
    if not force and getattr(port, 'distortion_information', None) is not None:
        # port has exhibit methods enabled
        return

    if type(port) == type:
        # set for the agg.Portfolio class - all objects new and existing
        logging.info('Enhancing agg.Portfolio class')

        def add_method(func):
            setattr(port, func.__name__, func)

    elif isinstance(port, Portfolio):
        # set for just one instance
        logging.info(f'Enhancing just this Portfolio instance, called {port.name}')

        def add_method(func):
            setattr(port, func.__name__, types.MethodType(func, port))

        def add_method(func):
            setattr(port, func.__name__, types.MethodType(func, port))

    else:
        raise ValueError(f'Item of type {type(port)} unexpected passed to add_enhanced_exhibit_methods. '
                         'Expecting agg.Portfolio class or an instance.')

    port.line_name_pipe = "|".join(port.line_names_ex)

    # namer helper classes
    port.premium_capital_renamer = {
        'Assets': "0. Assets",
        'T.A': '1. Allocated assets',
        'T.P': '2. Market value liability',
        'T.L': '3. Expected incurred loss',
        'T.M': '4. Margin',
        'T.LR': '5. Loss ratio',
        'T.Q': '6. Allocated equity',
        'T.ROE': '7. Cost of allocated equity',
        'T.PQ': '8. Premium to surplus ratio',
        'EPD': '9. Expected pol holder deficit'
    }

    def basic_loss_statistics(self, p=0.995, lines=None, line_names=None, deets=True):
        """
        mean, CV, skew, curt and some percentiles
        optionally add additional p quantiles

        lines = include not these lines to right with names as given

        """
        cols = ['Mean', 'EmpMean', 'MeanErr', 'CV', 'EmpCV', 'CVErr', 'Skew', 'EmpKurt', 'P99.0', 'P99.6']
        if not deets:
            for i in ['EmpMean', 'MeanErr', 'EmpCV', 'CVErr']:
                cols.remove(i)
        bls = self.audit_df[cols].T
        if p not in [.99, .996]:
            bls.loc[f'P{100 * p:.4f}', :] = \
                [ag.q(p) for ag in self.agg_list] + [self.q(p)]

        if lines:
            if line_names is None:
                line_names = [f'Not {line}' for line in lines]
            for line, line_name in zip(lines, line_names):
                # add not line which becomes the end of period reserves
                xs = self.density_df.loss
                ps = self.density_df[f'ημ_{line}']
                t = xs * ps
                ex1 = np.sum(t)
                t *= xs
                ex2 = np.sum(t)
                t *= xs
                ex3 = np.sum(t)
                t *= xs
                ex4 = np.sum(t)
                m, cv, s = agg.MomentAggregator.static_moments_to_mcvsk(ex1, ex2, ex3)
                # empirical kurtosis
                kurt = (ex4 - 4 * ex3 * ex1 + 6 * ex1 ** 2 * ex2 - 3 * ex1 ** 4) / ((m * cv) ** 4) - 3
                ans = np.zeros(3)
                temp = ps.cumsum()
                for i, p in enumerate([0.99, 0.995, p]):
                    ans[i] = (temp > p).idxmax()
                newcol = [m, cv, s, kurt] + list(ans)
                bls[line_name] = newcol[:len(bls)]

        self.EX_basic_loss_statistics = bls.rename(index=dict(EmpKurt='Kurt'),
                                                   columns=self.line_renamer)

    add_method(basic_loss_statistics)

    def distortion_information(self):
        """
        summary of the distortion calibration information
        """
        self.EX_distortion_information = self.dist_ans.reset_index(drop=False). \
            sort_values('method')[['method', 'param']]. \
            rename(columns=dict(method='Distortion', param='Shape Parameter')). \
            set_index('Distortion').rename(index=agg.Distortion._distortion_names_)

    add_method(distortion_information)

    def distortion_calibration(self):

        """
        one line summary from the distortion calibration
        was premium_capital_summary
        """

        self.EX_distortion_calibration = self.dist_ans.xs(self.distortion.name, level=2).iloc[:, :-1]
        self.EX_distortion_calibration = self.EX_distortion_calibration.reset_index(drop=False)
        self.EX_distortion_calibration.columns = ['$a$', 'LR', '$S(a)$', '$\\iota$', '$\\delta$', '$\\nu$',
                                                  '$EL(a)$', '$\\rho(X\\wedge a)$', 'Levg', '$\\bar Q(a)$',
                                                  'ROE', 'Shape']
        # delta and nu kinda useless
        self.EX_distortion_calibration = self.EX_distortion_calibration[
            ['Shape', '$a$', 'LR', '$S(a)$', '$\\iota$', '$\\nu$',
             '$EL(a)$', '$\\rho(X\\wedge a)$', 'Levg',
             '$\\bar Q(a)$', 'ROE']]
        self.EX_distortion_calibration.loc[0, 'Distortion'] = self.distortion.name
        self.EX_distortion_calibration = self.EX_distortion_calibration.set_index('Distortion')

    add_method(distortion_calibration)

    def premium_capital(self, a=0, p=0):
        """
        at a if given else p level of capital

        pricing story allows two asset levels...handle that with a concat

        was premium_capital_detail

        """
        a, p = self.set_a_p(a, p)

        if getattr(self, '_raw_premium_capital', None) is not None and self.last_a == a:
            # done already
            return

        # else recompute

        # story == run off
        # pricing report from adf
        dm = self.augmented_df.filter(regex=f'T.[MPQLROE]+.({self.line_name_pipe})').loc[[a]].T
        dm.index = dm.index.str.split('_', expand=True)
        self._raw_premium_capital = dm.unstack(1)
        self._raw_premium_capital = self._raw_premium_capital.droplevel(axis=1, level=0)  # .drop(index=['Assets'])
        self._raw_premium_capital.loc['T.A', :] = self._raw_premium_capital.loc['T.Q', :] + \
                                                  self._raw_premium_capital.loc['T.P', :]
        self._raw_premium_capital.index.name = 'Item'
        self.EX_premium_capital = self._raw_premium_capital. \
            rename(index=self.premium_capital_renamer, columns=self.line_renamer).sort_index()
        self.last_a = a

    add_method(premium_capital)

    def multi_premium_capital(self, As, keys=None):
        """
        concatenate multiple prem_capital exhibits

        """
        if keys is None:
            keys = [f'$a={i:.1f}$' for i in As]

        ans = []
        for a in As:
            self.premium_capital(a)
            ans.append(self.EX_premium_capital.copy())
        self.EX_multi_premium_capital = pd.concat(ans, axis=1, keys=keys, names=['Assets', "Line"])

    add_method(multi_premium_capital)

    def accounting_economic_balance_sheet(self, a=0, p=0):
        """
        story version assumes line 0 = reserves and 1 = prospective....other than that identical

        usual a and p rules
        """

        # check for update
        self.premium_capital(a, p)

        aebs = pd.DataFrame(0.0, index=self.line_names_ex + ['Assets', 'Equity'],
                            columns=['Statutory', 'Objective', 'Market', 'Difference'])
        slc = slice(0, len(self.line_names_ex))
        aebs.iloc[slc, 0] = self.audit_df['Mean'].values
        aebs.iloc[slc, 1] = self._raw_premium_capital.loc['T.L']
        aebs.iloc[slc, 2] = self._raw_premium_capital.loc['T.P']
        aebs.loc['Assets', :] = self._raw_premium_capital.loc['T.A', 'total']
        aebs.loc['Equity', :] = aebs.loc['Assets'] - \
                                aebs.loc['total']
        aebs[
            'Difference'] = aebs.Market - aebs.Objective
        # put in accounting order
        aebs = aebs.iloc[
            [-2] + list(range(len(self.line_names) + 1)) + [-1]]
        aebs.index.name = 'Item'
        self.EX_accounting_economic_balance_sheet = aebs.rename(index=self.line_renamer)

    add_method(accounting_economic_balance_sheet)

    def compare_allocations(self, verbose=False):

        """

        CAS-ASTIN talks...and general comparison of ROE, alloc capital etc.
        cycle round -

        """
        # utility
        if verbose:
            qd = lambda x: display(x.style)  # .format('{:.5g}'))
        else:
            qd = lambda x: 1

        # use last calibration / run
        dist_name = self.distortion.name
        a = self.ad_ans.audit_df.at['a', 'stat']

        display(HTML('<h3>Audit</h3>'))
        qd(self.ad_ans.audit_df)

        # double check
        a1 = float(self.ad_ans.audit_df.loc['a_cal'])
        if a1 != a:
            print('Warning: computed and input a values disagree\n' * 3, f'input {a}\ncalcd {a1}')

        # hack off exhibits
        display(HTML(f'<h3>Natural Allocation, $a={a:.1f}$</h3>'))
        bit = self.ad_ans.exhibit.loc[f'Dist {dist_name}']
        bit.index.name = None
        self.EX_natural_allocation_summary = bit.rename(
            index=dict(L='Loss', LR='Loss Ratio', M='Margin', P='Premium', PQ='P/S Ratio', Q='Equity'),
            columns=self.line_renamer).copy()
        self.EX_natural_allocation_summary.loc['Assets', :] = self.EX_natural_allocation_summary.loc['Premium'] + \
                                                              self.EX_natural_allocation_summary.loc['Equity']
        qd(self.EX_natural_allocation_summary)

        display(HTML(f'<h3>Allocated Capital Comparison, $a={a:.1f}$</h3>'))
        bit = self.ad_ans.exhibit.xs('Q', level=1)
        bit.index.name = None
        self.EX_allocated_capital_comparison = bit.rename(
            index={'T': f'Natural, {dist_name}'}, columns=self.line_renamer).copy()
        qd(self.EX_allocated_capital_comparison)

        display(HTML(f'<h3>Margin Comparison, $a={a:.1f}$</h3>'))
        bit = self.ad_ans.exhibit.xs('M', level=1)
        bit.index.name = None
        self.EX_margin_comparison = bit.rename(
            index={'T': f'Natural, {dist_name}'}, columns=self.line_renamer).copy()
        qd(self.EX_margin_comparison)

        display(HTML(f'<h3>Loss Ratio Comparison, $a={a:.1f}$</h3>'))
        bit = self.ad_ans.exhibit.xs('LR', level=1)
        bit.index.name = None
        self.EX_loss_ratio_comparison = bit.rename(
            index={'T': f'Natural, {dist_name}'}, columns=self.line_renamer).copy()
        qd(self.EX_loss_ratio_comparison)

        display(HTML(f'<h3>ROE Comparison, $a={a:.1f}$</h3>'))
        bit = self.ad_ans.exhibit.xs('ROE', level=1)
        bit.index.name = None
        self.EX_return_on_allocated_capital_comparison = bit.rename(
            index={'T': f'Natural, {dist_name}'}, columns=self.line_renamer).copy()
        qd(self.EX_return_on_allocated_capital_comparison)

    add_method(compare_allocations)

    # def cotvar(self, p, ROE): folded into portfolio object

    def make_all(self, p=0, a=0, As=None, ROE=0.1, mass_hints=None):
        """
        make all exhibits with sensible defaults
        if not entered, paid line is selected as the LAST line

        """
        a, p = self.set_a_p(a, p)

        self.basic_loss_statistics(p)

        # exhibits that require a distortion
        if self.distortion is not None:
            self.distortion_information()
            self.distortion_calibration()
            self.premium_capital(a=a, p=p)
            if As:
                self.multi_premium_capital(As)
            self.accounting_economic_balance_sheet(a=a, p=p)

    add_method(make_all)

    def show(self, fmt='{:.5g}'):
        """
        show all the made exhibits
        """
        display(HTML(f'<h2>Exhibits for {self.name.replace("_", " ").title()} Portfolio</h2>'))
        for x in dir(self):
            if x[0:3] == 'EX_':
                ob = getattr(self, x)
                if isinstance(ob, pd.DataFrame):
                    # which they all will be...
                    display(HTML(f'<h3>{x[3:].replace("_", " ").title()}</h3>'))
                    display(ob.style.format(fmt, subset=ob.select_dtypes(np.number).columns))
                    display(HTML(f'<hr>'))

    add_method(show)

    def profit_segment_plot(self, a, p, line_names, colors, transs, dist_name):
        """
        add all the lines, optionally translate
        requested distortion is applied on the fly

        """
        dist = self.dists[dist_name]
        col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for line, cn, trans in zip(line_names, colors, transs):
            c = col_list[cn]
            f1 = self.density_df[f'p_{line}'].cumsum()
            idx = (f1 < p) * (f1 > 1.0 - p)
            f1 = f1[idx]
            gf = 1 - dist.g(1 - f1)
            x = self.density_df.loss[idx] + trans
            a.plot(gf, x, '-', c=c, label=f'Risk Adj {line}' if trans == 0 else None)
            a.plot(f1, x, '--', c=c, label=line if trans == 0 else None)
            if trans == 0:
                a.fill_betweenx(x, gf, f1, color=c, alpha=0.5)
            else:
                a.fill_betweenx(x, gf, f1, color=c, edgecolor='black', alpha=0.5, hatch='+')
        a.set(ylim=[0, self.q(p)])
        a.legend(loc='upper left')

    add_method(profit_segment_plot)

    def natural_profit_segment_plot(self, a, p, line_names, colors, transs):
        """
        plot the natural allocations
        between 1-p and p th percentiles
        optionally translate a line
        works with augmented_df, no input dist

        """
        col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lw, up = self.q(1 - p), self.q(p)
        # common extract for all lines
        bit = self.augmented_df.query(f' {lw} <= loss <= {up} ')
        F = bit[f'F']
        gF = bit[f'gF']
        x = bit.loss
        for line, cn, trans in zip(line_names, colors, transs):
            c = col_list[cn]
            ser = bit[f'exeqa_{line}']
            a.plot(F, ser, ls='dashed', c=c)
            a.plot(gF, ser, c=c)
            if trans == 0:
                a.fill_betweenx(ser + trans, gF, F, color=c, alpha=0.5, label=line)
            else:
                a.fill_betweenx(ser, gF, F, color=c, alpha=0.5, label=line)
        a.set(ylim=[0, self.q(p)])
        a.legend(loc='upper left')
        # a.set(title=self.distortion)

    add_method(natural_profit_segment_plot)

    def density_sample(self, n=20, reg="loss|p_|exeqa_"):
        """
        sample of equally likely points from density_df with interesting columns
        reg - regex to select the columns
        """
        ps = np.linspace(0.001, 0.999, 20)
        xs = [self.q(i) for i in ps]
        return self.density_df.filter(regex=reg).loc[xs, :].rename(columns=self.renamer)

    add_method(density_sample)

    def biv_contour_plot(self, fig, ax, min_loss, max_loss, jump,
                         log=True, cmap='Greys', min_density=1e-15, levels=30, lines=None, linecolor='w',
                         colorbar=False, normalize=False, **kwargs):
        """
        Nake contour plot of line A vs line B
        Assumes port only has two lines

        Works with an extract density_df.loc[np.arange(min_loss, max_loss, jump), densities]
        (i.e., jump is the stride). Jump = 100 * bs is not bad...just think about how big the outer product will get!

        Param:

        min_loss  density for each line is sampled at min_loss:max_loss:jump
        max_loss
        jump
        log
        cmap
        min_density: smallest density to show on underlying log region; not used if log
        levels: number of contours or the actual contours if you like
        lines: iterable giving specific values of k to plot X+Y=k
        color_bar: show color bar
        normalize: if true replace Z with Z / sum(Z)
        kwargs passed to contourf (e.g., use for corner_mask=False, vmin,vmax)

        optionally could deal with diagonal lines, countour levels etc.
        originally from ch09
        """
        # careful about origin when big prob of zero loss
        npts = np.arange(min_loss, max_loss, jump)
        ps = [f'p_{i}' for i in self.line_names]
        bit = self.density_df.loc[npts, ps]
        n = len(bit)
        Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
        if normalize:
            Z = Z / np.sum(Z)
        X, Y = np.meshgrid(bit.index, bit.index)

        if log:
            z = np.log10(Z)
            mask = np.zeros_like(z)
            mask[z == -np.inf] = True
            mz = np.ma.array(z, mask=mask)
            cp = ax.contourf(X, Y, mz, levels=levels, cmap=cmap, **kwargs)
            # cp = ax.contourf(X, Y, mz, levels=np.linspace(-12, 0, levels), cmap=cmap, **kwargs)
            if colorbar:
                cb = fig.colorbar(cp, fraction=.1, shrink=0.5, aspect=10)
                cb.set_label('Log10(Density)')
        else:
            mask = np.zeros_like(Z)
            mask[Z < min_density] = True
            mz = np.ma.array(Z, mask=mask)
            cp = ax.contourf(X, Y, mz, levels=levels, cmap=cmap, **kwargs)
            if colorbar:
                cb = fig.colorbar(cp)
                cb.set_label('Density')
        # put in X+Y=c lines
        if lines is None:
            lines = np.arange(max_loss / 4, 2 * max_loss, max_loss / 4)
        try:
            for x in lines:
                ax.plot([0, x], [x, 0], lw=.75, c=linecolor, label=f'Sum = {x:,.0f}')
        except:
            pass

        title = (f'Bivariate Log Density Contour Plot\n{self.name.replace("_", " ").title()}'
                 if log
                 else f'Bivariate Density Contour Plot\n{self.name.replace("_", " ")}')

        # previously limits set from min_loss, but that doesn't seem right
        ax.set(xlabel=f'Line {self.line_names[0]}',
               ylabel=f'Line {self.line_names[1]}',
               xlim=[-max_loss / 50, max_loss],
               ylim=[-max_loss / 50, max_loss],
               title=title,
               aspect=1)

        # for post-processing
        self.X = X
        self.Y = Y
        self.Z = Z

    add_method(biv_contour_plot)

    def nice_program(self, wrap_col=90):
        """
        return wrapped version of port program
        :return:
        """
        return fill(self.program, wrap_col, subsequent_indent='\t\t', replace_whitespace=False)

    add_method(nice_program)

    def short_renamer(self, prefix='', postfix=''):
        if prefix:
            prefix = prefix + '_'
        if postfix:
            postfix = '_' + postfix

        knobble = lambda x: 'Total' if x == 'total' else x

        return {f'{prefix}{i}{postfix}': knobble(i).title() for i in self.line_names_ex}

    add_method(short_renamer)


def twelve_plot(self, fig, axs, p=0.999, p2=0.9999, xmax=0, ymax2=0, biv_log=True, legend_font=0,
                contour_scale=10, sort_order=None, kind='two', cmap='viridis'):
    """
    Twelve-up plot for ASTIN paper, by rc index:

    Greys for grey color map

    11 density
    12 log density
    13 biv density plot

    21 kappa
    22 alpha (from alpha beta plot 4)
    23 beta (?with alpha)

    row 3 = line A, row 4 = line B from alpha beta four 2
     1 S, gS, aS, bgS

    32 margin
    33 shift margin
    42 cumul margin
    43 natural profit compare

    Args
    ====
    self = portfolio or enhanced portfolio object
    p control xlim of plots via quantile; used if xmax=0
    p2 controls ylim for 33 and 34: stand alone M and natural M; used if ymax2=0
    biv_log - bivariate plot on log scale
    legend_font - fine tune legend font size if necessary
    sort_order = plot sorts by column and then .iloc[:, sort_order], if None [1,2,0]

    from common_scripts.py

    """

    a11, a12, a13, a21, a22, a23, a31, a32, a33, a41, a42, a43 = axs.flat
    col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lss = ['solid', 'dashed', 'dotted', 'dashdot']

    if sort_order is None:
        sort_order = [1, 2, 0]  # range(len(self.line_names_ex))

    if xmax == 0:
        xmax = self.q(p)
    ymax = xmax

    # density and log density

    temp = self.density_df.filter(regex='p_').rename(columns=self.short_renamer('p')).sort_index(axis=1).loc[:xmax]
    temp = temp.iloc[:, sort_order]
    temp.index.name = 'Loss'
    l1 = temp.plot(ax=a11, lw=1)
    l2 = temp.plot(ax=a12, lw=1, logy=True)
    l1.lines[-1].set(linewidth=1.5)
    l2.lines[-1].set(linewidth=1.5)
    a11.set(title='Density')
    a12.set(title='Log density')
    a11.legend()
    a12.legend()

    # biv den plot
    if kind == 'two':
        # biv den plot
        # min_loss, max_loss, jump = 0, xmax, (2 ** (self.log2 - 8)) * self.bs
        xmax = self.snap(xmax)
        min_loss, max_loss, jump = 0, xmax, self.snap(xmax / 255)
        min_density = 1e-15
        levels = 30
        color_bar = False

        # careful about origin when big prob of zero loss
        # handle for discrete distributions
        ps = [f'p_{i}' for i in self.line_names]
        title = 'Bivariate density'
        query = ' or '.join([f'`p_{i}` > 0' for i in self.line_names])
        if self.density_df.query(query).shape[0] < 512:
            logger.info('Contour plot has few points...going discrete...')
            bit = self.density_df.query(query)
            n = len(bit)
            Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
            X, Y = np.meshgrid(bit.index, bit.index)
            norm = mpl.colors.Normalize(vmin=-10, vmax=np.log10(np.max(Z.flat)))
            cm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper = cm.to_rgba
            a13.scatter(x=X.flat, y=Y.flat, s=1000 * Z.flatten(), c=mapper(np.log10(Z.flat)))
            # edgecolor='C2', lw=1, facecolors='none')
            a13.set(xlim=[min_loss - (max_loss - min_loss) / 10, max_loss],
                    ylim=[min_loss - (max_loss - min_loss) / 10, max_loss])

        else:
            npts = np.arange(min_loss, max_loss, jump)
            bit = self.density_df.loc[npts, ps]
            n = len(bit)
            Z = bit[ps[1]].to_numpy().reshape(n, 1) @ bit[ps[0]].to_numpy().reshape(1, n)
            Z = Z / np.sum(Z)
            X, Y = np.meshgrid(bit.index, bit.index)

            if biv_log:
                z = np.log10(Z)
                mask = np.zeros_like(z)
                mask[z == -np.inf] = True
                mz = np.ma.array(z, mask=mask)
                cp = a13.contourf(X, Y, mz, levels=np.linspace(-17, 0, levels), cmap=cmap)
                if color_bar:
                    cb = fig.colorbar(cp)
                    cb.set_label('Log10(Density)')
            else:
                mask = np.zeros_like(Z)
                mask[Z < min_density] = True
                mz = np.ma.array(Z, mask=mask)
                cp = a13.contourf(X, Y, mz, levels=levels, cmap=cmap)
                if color_bar:
                    cb = fig.colorbar(cp)
                    cb.set_label('Density')
            a13.set(xlim=[min_loss, max_loss], ylim=[min_loss, max_loss])

        # put in X+Y=c lines
        lines = np.arange(contour_scale / 4, 2 * contour_scale + 1, contour_scale / 4)
        logger.debug(f'Contour lines based on {contour_scale} gives {lines}')
        for x in lines:
            a13.plot([0, x], [x, 0], ls='solid', lw=.35, c='k', alpha=0.5, label=f'Sum = {x:,.0f}')

        a13.set(xlabel=f'Line {self.line_names[0]}',
                ylabel=f'Line {self.line_names[1]}',
                title=title,
                aspect=1)
    else:
        # kind == 'three' plot survival functions...bivariate plots elsewhere
        l1 = (1 - temp.cumsum()).plot(ax=a13, lw=1)
        l1.lines[-1].set(linewidth=1.5)
        a13.set(title='Survival Function')
        a13.legend()

    # kappa
    bit = self.density_df.loc[:xmax]. \
        filter(regex=f'^exeqa_({self.line_name_pipe})$')
    bit = bit.iloc[:, sort_order]
    bit.rename(columns=self.short_renamer('exeqa')).replace(0, np.nan). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a21, lw=1)
    a21.set(title='$\\kappa_i(x)=E[X_i\\mid X=x]$')
    # ugg for ASTIN paper example
    # a21.set(xlim=[0,5], ylim=[0,5])
    a21.set(xlim=[0, xmax], ylim=[0, xmax], aspect='equal')
    a21.legend(loc='upper left')

    # alpha and beta
    aug_df = self.augmented_df
    aug_df.filter(regex=f'exi_xgta_({self.line_name_pipe})'). \
        rename(columns=self.short_renamer('exi_xgta')). \
        sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a22, lw=1)
    for ln, ls in zip(a22.lines, lss[1:]):
        ln.set_linestyle(ls)
    a22.legend()
    a22.set(xlim=[0, xmax], title='$\\alpha_i(x)=E[X_i/X\\mid X>x]$');

    bit = aug_df.query(f'loss < {xmax}').filter(regex=f'exi_xgtag?_({self.line_name_pipe})')
    bit.rename(columns=self.short_renamer('exi_xgtag')). \
        sort_index(axis=1).plot(ylim=[-0.05, 1.05], ax=a23)
    for i, l in enumerate(a23.lines[len(self.line_names):]):
        if l.get_label()[0:3] == 'exi':
            a23.lines[i].set(linewidth=2, ls=lss[1 + i])
            l.set(color=f'C{i}', linestyle=lss[1 + i], linewidth=1,
                  alpha=.5, label=None)
    a23.legend(loc='upper left');
    a23.set(xlim=[0, xmax], title="$\\beta_i(x)=E_{Q}[X_i/X \\mid X> x]$");

    aug_df.filter(regex='M.M').rename(columns=self.short_renamer('M.M')). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a32, lw=1)
    a32.set(xlim=[0, xmax], title='Margin density $M_i(x)$')

    aug_df.filter(regex='T.M').rename(columns=self.short_renamer('T.M')). \
        sort_index(axis=1).iloc[:, sort_order].plot(ax=a42, lw=1)
    a42.set(xlim=[0, xmax], title='Margin $\\bar M_i(x)$');

    # by line S, gS, aS, bgS
    adf = self.augmented_df.loc[:xmax]
    if kind == 'two':
        zipper = zip(range(2), sorted(self.line_names), [a31, a41])
    else:
        zipper = zip(range(3), sorted(self.line_names), [a31, a41, a33])
    for i, line, a in zipper:
        a.plot(adf.loss, adf.S, c=col_list[2], ls=lss[1], lw=1, alpha=0.5, label='$S$')
        a.plot(adf.loss, adf.gS, c=col_list[2], ls=lss[0], lw=1, alpha=0.5, label='$g(S)$')
        a.plot(adf.loss, adf.S * adf[f'exi_xgta_{line}'], c=col_list[i], ls=lss[1], lw=1, label=f'$\\alpha S$ {line}')
        a.plot(adf.loss, adf.gS * adf[f'exi_xgtag_{line}'], c=col_list[i], ls=lss[0], lw=1,
               label=f"$\\beta g(S)$ {line}")
        a.set(xlim=[0, ymax])

        a.set(title=f'Line = {line}')
        a.legend()
        a.set(xlim=[0, ymax])
        a.legend(loc='upper right')

    alpha = 0.05
    if kind == 'two':
        # three mode this is used for the third line
        # a33 from profit segment plot
        # special ymax for last two plots
        ymax = ymax2 if ymax2 > 0 else self.q(p2)
        # if could have changed
        p2 = self.cdf(ymax)
        for cn, ln in enumerate(sort_order):
            line = sorted(self.line_names_ex)[ln]
            c = col_list[cn]
            s = lss[cn]
            # print(line, s)
            f1 = self.density_df[f'p_{line}'].cumsum()
            idx = (f1 < p2) * (f1 > 1.0 - p2)
            f1 = f1[idx]
            gf = 1 - self.distortion.g(1 - f1)
            x = self.density_df.loss[idx]
            a33.plot(gf, x, c=c, ls=s, lw=1, label=None)
            a33.plot(f1, x, ls=s, c=c, lw=1, label=None)
            # a33.plot(f1, x, ls=lss[1], c=c, lw=1, label=None)
            a33.fill_betweenx(x, gf, f1, color=c, alpha=alpha, label=line.title())
        a33.set(ylim=[0, ymax], title='Stand-alone $M$')
        a33.legend(loc='upper left')  # .set_visible(False)

    # a43 from natural profit segment plot
    # common extract for all lines
    lw, up = self.q(1 - p2), ymax
    bit = self.augmented_df.query(f' {lw} <= loss <= {up} ')
    F = bit[f'F']
    gF = bit[f'gF']
    # x = bit.loss
    for cn, ln in enumerate(sort_order):
        line = sorted(self.line_names_ex)[ln]
        c = col_list[cn]
        s = lss[cn]
        ser = bit[f'exeqa_{line}']
        if kind == 'three':
            a43.plot(1 / (1 - F), ser, lw=1, ls=lss[1], c=c)
            a43.plot(1 / (1 - gF), ser, lw=1, c=c)
            a43.set(xlim=[1, 1e4], xscale='log')
            a43.fill_betweenx(ser, 1 / (1 - gF), 1 / (1 - F), color=c, alpha=alpha, lw=0.5, label=line.title())
        else:
            a43.plot(F, ser, lw=1, ls=s, c=c)
            a43.plot(gF, ser, lw=1, ls=s, c=c)
            a43.fill_betweenx(ser, gF, F, color=c, alpha=alpha, lw=0.5, label=line.title())
    a43.set(ylim=[0, ymax], title='Natural $M$')
    a43.legend(loc='upper left')  # .set_visible(False)

    if legend_font:
        for ax in axs.flat:
            try:
                if ax is not a13:
                    ax.legend(prop={'size': 7})
            except:
                pass


def pricing(port, p, roe, as_dataframe=True):
    """
    Make nice stats output for pricing
    from common_scripts.py
    """
    a = port.q(p)
    loss = port.density_df.loc[a, 'lev_total']
    v = 1 / (1 + roe)
    d = 1 - v
    premium = loss * v + d * a
    q = a - premium
    ans = pd.Series([loss, premium - loss, premium, loss / premium, q, (premium - loss) / q, a, premium / q],
                    index=['L', 'M', 'P', 'LR', 'Q', 'ROE', 'a', 'PQ'])
    if as_dataframe:
        df = pd.DataFrame(ans, columns=['total'])
        df.index.name = 'stat'
        return df
    else:
        return ans


def bivariate_density_plots(axi, ports, xmax, contour_scale, biv_log=True, cmap='viridis',
                            levels=30, color_bar=False):
    """
    bivarate plots of each line against the others for each portfolio in case
    arguments as for twelve_plot / bivden plot

    axi = iterator with enough exes
    ports = iterable of ports (list, dict.values(), etc.)


    from common_scripts.py
    """

    min_density = 1e-15
    # careful about origin when big prob of zero loss
    # handle for discrete distributions
    for port in ports:
        # this can fail to create enough points (e.g. big log2), you lose resolution at left hand end
        # min_loss, max_loss, jump = 0, xmax, (2 ** (port.log2 - 8)) * port.bs
        # this version ensures you get a reasonable number of points .... you still have to pick xmax carefullly
        xmax = port.snap(xmax)
        min_loss, max_loss, jump = 0, xmax, port.snap(xmax / 255)
        ps = [f'p_{i}' for i in port.line_names]
        title = port.name
        query = ' or '.join([f'`p_{i}` > 0' for i in port.line_names])
        nobs = port.density_df.query(query).shape[0]
        nlines = len(port.line_names)
        if nobs < 512:
            logger.info('Contour plot has few points...going discrete...')
            bit = port.density_df.query(query)
            n = len(bit)
            for i, j in product(range(nlines), range(nlines)):
                if i < j:
                    a13 = next(axi)
                    Z = bit[ps[j]].to_numpy().reshape(n, 1) @ bit[ps[i]].to_numpy().reshape(1, n)
                    X, Y = np.meshgrid(bit.index, bit.index)
                    norm = mpl.colors.Normalize(vmin=-10, vmax=np.log10(np.max(Z.flat)))
                    cm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis')
                    mapper = cm.to_rgba
                    a13.scatter(x=X.flat, y=Y.flat, s=1000 * Z.flatten(), c=mapper(np.log10(Z.flat)))
                    # edgecolor='C2', lw=1, facecolors='none')
                    a13.set(xlim=[min_loss - (max_loss - min_loss) / 10, max_loss],
                            ylim=[min_loss - (max_loss - min_loss) / 10, max_loss])
        else:
            npts = np.arange(min_loss, max_loss, jump)
            bit = port.density_df.loc[npts, ps]
            n = len(bit)
            for i, j in product(range(nlines), range(nlines)):
                if i < j:
                    a13 = next(axi)
                    Z = bit[ps[j]].to_numpy().reshape(n, 1) @ bit[ps[i]].to_numpy().reshape(1, n)
                    Z = Z / np.sum(Z)
                    X, Y = np.meshgrid(bit.index, bit.index)

                    if biv_log:
                        z = np.log10(Z)
                        mask = np.zeros_like(z)
                        mask[z == -np.inf] = True
                        mz = np.ma.array(z, mask=mask)
                        cp = a13.contourf(X, Y, mz, levels=np.linspace(-17, 0, levels), cmap=cmap)
                        if color_bar:
                            cb = a13.figure.colorbar(cp)
                            cb.set_label('Log10(Density)')
                    else:
                        mask = np.zeros_like(Z)
                        mask[Z < min_density] = True
                        mz = np.ma.array(Z, mask=mask)
                        cp = a13.contourf(X, Y, mz, levels=levels, cmap=cmap)
                        if color_bar:
                            cb = a13.figure.colorbar(cp)
                            cb.set_label('Density')
                    a13.set(xlim=[min_loss, max_loss], ylim=[min_loss, max_loss])

                    # put in X+Y=c lines
                    lines = np.arange(contour_scale / 4, 2 * contour_scale + 1, contour_scale / 4)
                    logger.debug(f'Contour lines based on {contour_scale} gives {lines}')
                    for x in lines:
                        a13.plot([0, x], [x, 0], ls='solid', lw=.35, c='k', alpha=0.5, label=f'Sum = {x:,.0f}')

                    a13.set(xlabel=f'Line {port.line_names[i]}',
                            ylabel=f'Line {port.line_names[j]}',
                            title=title,
                            aspect=1)


class Bounds(object):
    """
    Implement IME pricing bounds methodology.

    Typical usage:

    * Create a Portfolio or Aggregate object a
    * bd = cd.Bounds(a)
    * bd.tvar_cloud('line', premium=, a=, n_tps=, s=, kind=)
    * p_star = bd.p_star('line', premium)
    * bd.cloud_view(axes, ...)

    distribution_spec = Portfolio or Portfolio.density_df dataframe or pd.Series (must have loss as index)
    If DataFrame or Series values interpreted as desnsity, sum to 1. F, S, exgta all computed using Portfolio
    methdology
    If DataFrame line --> p_{line}

    from common_scripts.cs
    """

    def __init__(self, distribution_spec):
        assert isinstance(distribution_spec, (pd.Series, pd.DataFrame, agg.Portfolio, agg.Aggregate))
        self.distribution_spec = distribution_spec
        # although passed as input to certain functions (tvar with bounds) b is actually fixed
        self.b = 0
        self.Fb = 0
        self.tvar_function = None
        self.tvars = None
        self.tps = None
        self.weight_df = None
        self.idx = None
        self.hinges = None
        # in cases where we hold the tvar function here
        self._tail_var = None
        self._inverse_tail_var = None
        self.cloud_df = None
        # uniform mode
        self._t_mode = 'u'
        # data frame with tvar weights and principal extreme distortion weights by method
        self.pedw_df = None
        self._tvar_df = None
        # hack for beta distribution, you want to force 1 to be in tvar ps, but Fp = 1
        # TODO figure out why p_star grinds to a halt if you input b < inf
        self.add_one = False

    def __repr__(self):
        """
        gets called automatically but so we can tweak
        :return:
        """
        return 'My Bounds Object at ' + super(Bounds, self).__repr__()

    def __str__(self):
        return 'Hello' + super(Bounds, self).__repr__()

    @property
    def tvar_df(self):
        if self._tvar_df is None:
            self._tvar_df = pd.DataFrame({'p': self.tps, 'tvar': self.tvars}).set_index('p')
        return self._tvar_df

    @property
    def t_mode(self):
        return self._t_mode

    @t_mode.setter
    def t_mode(self, val):
        assert val in ['u', 'gl']
        self._t_mode = val

    def make_tvar_function(self, line, b=np.inf):
        """
        make the tvar function from a Series p_total indexed by loss
        Includes determining sup and putting in value for zero
        If sup is largest value in index, sup set to inf

        also sets self.Fb

        Applies to min(Line, b)

        :param line:
        :param b:  bound on the losses, e.g., to model limited liability insurer
        :return:
        """
        self.b = b
        if isinstance(self.distribution_spec, agg.Portfolio):
            assert line in self.distribution_spec.line_names_ex
            if line == 'total':
                self.tvar_function = self.distribution_spec.tvar
                self.Fb = self.distribution_spec.cdf(b)
            else:
                ag = getattr(self.distribution_spec, line)
                self.tvar_function = ag.tvar
                self.Fb = ag.cdf(b)
            if np.isinf(b): self.Fb = 1.0
            return

        elif isinstance(self.distribution_spec, agg.Aggregate):
            self.tvar_function = self.distribution_spec.tvar
            self.Fb = self.distribution_spec.cdf(b)
            if np.isinf(b): self.Fb = 1.0
            return

        elif isinstance(self.distribution_spec, pd.DataFrame):
            assert f'p_{line}' in self.distribution_spec.columns
            # given a port.density_df
            p_total = self.distribution_spec[f'p_{line}']

        elif isinstance(self.distribution_spec, pd.Series):
            logger.info('tvar_array using Series')
            p_total = self.distribution_spec

        # need to create tvar function on the fly, using same method as Portfolio and Aggregate:
        bs = p_total.index[1]
        F = p_total.cumsum()
        if np.isinf(b):
            self.Fb = 0
        else:
            self.Fb = F[b]

        S = p_total.shift(-1, fill_value=min(p_total.iloc[-1], max(0, 1. - (p_total.sum()))))[::-1].cumsum()[::-1]
        lev = S.shift(1, fill_value=0).cumsum() * bs
        ex1 = lev.iloc[-1]
        ex = np.sum(p_total * p_total.index)
        logger.info(f'Computed mean loss for {line} = {ex:,.15f} (diff {ex - ex1:,.15f}) max F = {max(F)}')
        exgta = (ex - lev) / S + S.index
        sup = (p_total[::-1] > 0).idxmax()
        if sup == p_total.index[-1]:
            sup = np.inf
        exgta[S == 0] = sup
        logger.info(f'sup={sup}, max = {(p_total[::-1] > 0).idxmax()} "inf" = {p_total.index[-1]}')

        def _tvar(p, kind='interp'):
            """
            UNLIMITED tvar function!
            :param p:
            :param kind:
            :return:
            """
            if kind == 'interp':
                # original implementation interpolated
                if self._tail_var is None:
                    # make tvar function
                    self._tail_var = interp1d(F, exgta, kind='linear', bounds_error=False,
                                              fill_value=(0, sup))
                return self._tail_var(p)
            elif kind == 'inverse':
                if self._inverse_tail_var is None:
                    # make tvar function
                    self._inverse_tail_var = interp1d(exgta, F, kind='linear', bounds_error=False,
                                                      fill_value='extrapolate')
                return self._inverse_tail_var(p)

        self.tvar_function = _tvar

    def make_ps(self, n, mode):
        """
        mode are you making s points (always uniform) or tvar p points (use t_mode)?
        self.t_mode == 'u': make uniform s points against which to evaluate g from 0 to 1 inclusive with more around 0
        self.t_mode == 'gl': make Gauss-Legndre p points at which TVaRs are evaluated from 0 inclusive to 1 exclusive with more around 1

        :param n:
        :return:
        """
        assert mode in ('s', 't')

        if mode == 't' and (self.Fb < 1 or self.add_one):
            # we will add 1 at the end
            n -= 1

        # Gauus Legendre points
        lg = np.polynomial.legendre.leggauss

        if self.t_mode == 'gl':
            if mode == 's':
                x, wts = lg(n - 2)
                ps = np.hstack((0, (x + 1) / 2, 1))

            elif mode == 't':
                x, wts = lg(n * 2 + 1)
                ps = x[n:]
        elif self.t_mode == 'u':
            if mode == 's':
                ps = np.linspace(1 / n, 1, n)
            elif mode == 't':
                # exclude 1 (sup distortion) at the end; 0=mean
                ps = np.linspace(0, 1, n, endpoint=False)
        # always ensure that 1  is in ps for t mode when b < inf if Fb < 1
        if mode == 't' and self.Fb < 1 or self.add_one:
            ps = np.hstack((ps, 1))
        return ps

    def tvar_array(self, line, n_tps=256, b=np.inf, kind='interp'):
        """
        compute tvars at n equally spaced points, tps


        :param line:
        :param n_tps:  number of tvar p points, default 256
        :param b: cap on losses applied before computing TVaRs (e.g., adjust losses for finite assets b).
        Use np.inf for unlimited losses.
        :param kind: if interp  uses the standard function, easy, for continuous distributions; if 'tail' uses
        explicit integration of tail values, for discrete distributions
        :return:
        """
        assert kind in ('interp', 'tail')
        self.make_tvar_function(line, b)

        logger.info(f'F(b) = {self.Fb:.5f}')
        # tvar p values should linclude 0 (the mean) but EXCLUDE 1
        # self.tps = np.linspace(0.5 / n_tps, 1 - 0.5 / n_tps, n_tps)
        self.tps = self.make_ps(n_tps, 't')

        if kind == 'interp':
            self.tvars = self.tvar_function(self.tps)
            if not np.isinf(b):
                # subtract S(a)(TVaR(F(a)) - a)
                # do all at once here - do not call self.tvar_with_bounds function
                self.tvars = np.where(self.tps <= self.Fb,
                                      self.tvars - (1 - self.Fb) * (self.tvar_function(self.Fb) - b) / (1 - self.tps),
                                      b)
        elif kind == 'tail':
            self.tvars = np.array([self.tvar_with_bound(i, b, kind) for i in self.tps])

    def p_star(self, line, premium, b=np.inf, kind='interp'):
        """
        Compute p* so TVaR @ p* of min(X, b) = premium

        In this case the cap b has an impact (think of integrating q(p) over p to 1, q is impacted by b)

        premium <= b is required (no rip off condition)

        If b < inf then must solve TVaR(p) - (1 - F(b)) / (1 - p)[TVaR(F(b)) - b] = premium
        Let k = (1 - F(b)) [TVaR(F(b)) - b], so solving

        f(p) = TVaR(p) - k / (1 - p) - premium == 0

        using NR

        :param line:
        :param premium: target premium
        :param b:  bound
        :return:
        """
        assert kind in ('interp', 'tail')
        if premium > b:
            raise ValueError(f'p_star must have premium ({premium}) <= largest loss bound ({b})')

        if kind == 'interp':

            self.make_tvar_function(line, b)

            if np.isinf(b):
                p_star = self.tvar_function(premium, 'inverse')
            else:
                # nr, remember F(a) is self.Fa set by make_tvar_function
                k = (1 - self.Fb) * (self.tvar_function(self.Fb) - b)

                def f(p):
                    return self.tvar_function(p) - k / (1 - p) - premium

                # should really compute f' numerically, but...
                fp = 100
                p = 0.5
                iters = 0
                delta = 1e-10
                while abs(fp) > 1e-6 and iters < 20:
                    fp = f(p)
                    fpp = (f(p + delta) - f(p)) / delta
                    pnew = p - fp / fpp
                    if 0 <= pnew <= 1:
                        p = pnew
                    elif pnew < 0:
                        p = p / 2
                    else:
                        #  pnew > 1:
                        p = (1 + p) / 2

                if iters == 20:
                    logger.warning(f'Questionable convergence solving for p_star, last error {fp}.')
                p_star = p

        elif kind == 'tail':
            def f(p):
                return self.tvar_with_bound(p, b, 'tail') - premium

            fp = 100
            p = 0.5
            iters = 0
            delta = 1e-10
            while abs(fp) > 1e-6 and iters < 20:
                fp = f(p)
                fpp = (f(p + delta) - f(p)) / delta
                pnew = p - fp / fpp
                if 0 <= pnew <= 1:
                    p = pnew
                elif pnew < 0:
                    p = p / 2
                else:
                    #  pnew > 1:
                    p = (1 + p) / 2

            if iters == 20:
                logger.warning(f'Questionable convergence solving for p_star, last error {fp}.')
            p_star = p

        return p_star

    def tvar_with_bound(self, p, b=np.inf, kind='interp'):
        """
        compute tvar taking bound into account
        assumes tvar_function setup

        Warning: b must equal the b used when calibrated. The issue is computing F
        varies with the type of underlying portfolio. This is fragile.
        Added storing b and checking equal. For backwards comp. need to keep b argument

        :param p:
        :param b:
        :return:
        """
        assert self.tvar_function is not None
        assert b == self.b

        if kind == 'interp':
            tvar = self.tvar_function(p)
            if not np.isinf(b):
                if p < self.Fb:
                    tvar = tvar - (1 - self.Fb) * (self.tvar_function(self.Fb) - b) / (1 - p)
                else:
                    tvar = b
        elif kind == 'tail':
            # use the tail method for discrete distributions
            tvar = self.distribution_spec.tvar(p, 'tail')
            if not np.isinf(b):
                if p < self.Fb:
                    tvar = tvar - (1 - self.Fb) * (self.distribution_spec.tvar(self.Fb, 'tail') - b) / (1 - p)
                else:
                    tvar = b
        return tvar

    def compute_weight(self, premium, p0, p1, b=np.inf, kind='interp'):
        """
        compute the weight for a single TVaR p0 < p1 value pair

        :param line:
        :param premium:
        :param tp:
        :param b:
        :return:
        """

        assert p0 < p1
        assert self.tvar_function is not None

        lhs = self.tvar_with_bound(p0, b, kind)
        rhs = self.tvar_with_bound(p1, b, kind)

        assert lhs != rhs
        weight = (premium - lhs) / (rhs - lhs)
        return weight

    def compute_weights(self, line, premium, n_tps, b=np.inf, kind='interp'):
        """
        Compute the weights of the extreme distortions

        Applied to min(line, b)  (allows to work for net)

        Note: independent of the asset level

        :param line: within port, or total
        :param premium: target premium for the line
        :param n_tps: number of tvar p points (tps)number of tvar p points (tps)number of tvar p points
            (tps)number of tvar p points (tps).
        :param b: loss bound: compute weights for min(line, b); generally used for net losses only.
        :return:
        """

        self.tvar_array(line, n_tps, b, kind)
        # you add zero, so there will be one additional point
        # n_tps += 1
        p_star = self.p_star(line, premium, b, kind)
        if p_star in self.tps:
            logger.critical('p_star in tps')
            # raise ValueError()

        lhs = self.tps[self.tps <= p_star]
        rhs = self.tps[self.tps > p_star]

        tlhs = self.tvars[self.tps <= p_star]
        trhs = self.tvars[self.tps > p_star]

        lhs, rhs = np.meshgrid(lhs, rhs)
        tlhs, trhs = np.meshgrid(tlhs, trhs)

        df = pd.DataFrame({'p_lower': lhs.flat, 'p_upper': rhs.flat,
                           't_lower': tlhs.flat, 't_upper': trhs.flat,
                           })
        # will fail when p_star in self.ps; let's deal with then when it happens
        df['weight'] = (premium - df.t_lower) / (df.t_upper - df.t_lower)

        df = df.set_index(['p_lower', 'p_upper'], verify_integrity=True)
        df = df.sort_index()

        if p_star in self.tps:
            # raise ValueError('Found pstar in ps')
            logger.critical(f'Found p_star = {p_star} in ps!!')
            df.at[(p_star, p_star), 'weight'] = 1.0

        logger.info(f'p_star={p_star:.4f}, len(p<=p*) = {len(df.index.levels[0])}, '
                    f'len(p>p*) = {len(df.index.levels[1])}; '
                    f' pstar in ps: {p_star in self.tps}')

        self.weight_df = df

        # index for tp values
        r = np.arange(n_tps)
        r_rhs, r_lhs = np.meshgrid(r[self.tps > p_star], r[self.tps <= p_star])
        self.idx = np.vstack((r_lhs.flat, r_rhs.flat)).reshape((2, r_rhs.size))

    def tvar_hinges(self, s):
        """
        make the tvar hinge functions by evaluating each tvar_p(s) = min(1, s/(1-p) for p in tps, at EP points s

        all arguments in [0,1] x [0,1]

        :param s:
        :return:
        """

        self.hinges = coo_matrix(np.minimum(1.0, s.reshape(1, len(s)) / (1.0 - self.tps.reshape(len(self.tps), 1))))

    def tvar_cloud(self, line, premium, a, n_tps, s, kind='interp'):
        """
        weight down tvar functions to the extremal convex measures

        asset level a acts like an agg stop on what is being priced, i.e. we are working with min(X, a)

        :param line:
        :param premium:
        :param a:
        :param n_tps:
        :param s:
        :param b:  bound, applies to min(line, b)
        :return:
        """

        self.compute_weights(line, premium, n_tps, a, kind)

        if type(s) == int:
            # points at which g is evaluated - all OK to include 0 and 1
            # s = np.linspace(0, 1, s+1, endpoint=True)
            s = self.make_ps(s, 's')

        self.tvar_hinges(s)

        ml = coo_matrix((1 - self.weight_df.weight, (np.arange(len(self.weight_df)), self.idx[0])),
                        shape=(len(self.weight_df), len(self.tps)))
        mr = coo_matrix((self.weight_df.weight, (np.arange(len(self.weight_df)), self.idx[1])),
                        shape=(len(self.weight_df), len(self.tps)))
        m = ml + mr

        logger.info(f'm shape = {m.shape}, hinges shape = {self.hinges.shape}, types {type(m)}, {type(self.hinges)}')

        self.cloud_df = pd.DataFrame((m @ self.hinges).T.toarray(), index=s, columns=self.weight_df.index)
        self.cloud_df.index.name = 's'

    def cloud_view(self, axs, n_resamples, scale='linear', alpha=0.05, pricing=True, distortions=None,
                   title='', lim=(-0.025, 1.025), check=False):
        """
        visualize the cloud with n_resamples

        after you have recomputed...

        if there are distortions plot on second axis

        :param axs:
        :param n_resamples: if random sample
        :param scale: linear or return
        :param alpha: opacity
        :param pricing: restrict to p_max = 0, ensuring g(s)<1 when s<1
        :param distortions:
        :param title: optional title (applied to all plots)
        :param lim: axis limits
        :param check:   construct and plot Distortions to check working ; reduces n_resamples to 5
        :return:
        """
        assert scale in ['linear', 'return']
        assert not distortions or (len(axs.flat) > 1)
        bit = None
        if check: n_resamples = min(n_resamples, 5)
        norm = mpl.colors.Normalize(0, 1)
        cm = mpl.cm.ScalarMappable(norm=norm, cmap='viridis_r')
        mapper = cm.get_cmap()

        def plot_max_min(ax):
            ax.fill_between(self.cloud_df.index, self.cloud_df.min(1), self.cloud_df.max(1), facecolor='C7', alpha=.15)
            self.cloud_df.min(1).plot(ax=ax, label='_nolegend_', lw=1, ls='-', c='k')
            self.cloud_df.max(1).plot(ax=ax, label="_nolegend_", lw=1, ls='-', c='k')

        logger.info('starting cloudview...')
        if scale == 'linear':
            ax = axs[0]
            if n_resamples > 0:
                if pricing:
                    if n_resamples < 10:
                        bit = self.weight_df.xs(0, drop_level=False).sample(n=n_resamples, replace=True).reset_index()
                    else:
                        bit = self.weight_df.xs(0, drop_level=False).reset_index()
                else:
                    bit = self.weight_df.sample(n=n_resamples, replace=True).reset_index()
                logger.info('cloudview...done 1')
                # display(bit)
                for i in bit.index:
                    pl, pu, tl, tu, w = bit.loc[i]
                    self.cloud_df[(pl, pu)].plot(ax=ax, lw=1, c=mapper(w), alpha=alpha, label=None)
                    if check:
                        # put in actual for each sample
                        d = agg.Distortion('wtdtvar', w, df=[pl, pu])
                        gs = d.g(s)
                        ax.plot(s, gs, c=mapper(w), lw=2, ls='--', alpha=.5, label=f'ma ({pl:.3f}, {pu:.3f}) ')
                ax.get_figure().colorbar(cm, ax=ax, shrink=.5, aspect=16, label='Weight to Higher Threshold')
            else:
                logger.info('cloudview: no resamples, skipping 1')
            logger.info('cloudview: start max/min')
            plot_max_min(ax)
            logger.info('cloudview: done with max/min')
            for ln in ax.lines:
                ln.set(label=None)
            if check:
                ax.legend(loc='lower right', fontsize='large')
            ax.plot([0, 1], [0, 1], c='k', lw=.25, ls='-')
            ax.set(xlim=lim, ylim=lim, aspect='equal')

            if type(distortions) == dict:
                distortions = [distortions]
            if distortions == 'space':
                ax = axs[1]
                plot_max_min(ax)
                ax.plot([0, 1], [0, 1], c='k', lw=.25, ls='-', label='_nolegend_')
                ax.legend(loc='lower right', ncol=3, fontsize='large')
                ax.set(xlim=lim, ylim=lim, aspect='equal')
            elif type(distortions) == list:
                logger.info('cloudview: start 4 adding distortions')
                name_mapper = {'roe': 'CCoC', 'tvar': 'TVaR(p*)', 'ph': 'PH', 'wang': 'Wang', 'dual': 'Dual'}
                s = np.linspace(0, 1, 1001)
                lss = list(mpl.lines.lineStyles.keys())
                for ax, dist_dict in zip(axs[1:], distortions):
                    ii = 1
                    for k, d in dist_dict.items():
                        gs = d.g(s)
                        k = name_mapper.get(k, k)
                        ax.plot(s, gs, lw=1, ls=lss[ii], label=k)
                        ii += 1
                    plot_max_min(ax)
                    ax.plot([0, 1], [0, 1], c='k', lw=.25, ls='-', label='_nolegend_')
                    ax.legend(loc='lower right', ncol=3, fontsize='large')
                    ax.set(xlim=lim, ylim=lim, aspect='equal')
            else:
                # do nothing
                pass

        elif scale == 'return':
            ax = axs[0]
            bit = self.cloud_df.sample(n=n_resamples, axis=1)
            bit.index = 1 / bit.index
            bit = 1 / bit
            bit.plot(ax=ax, lw=.5, c='C7', alpha=alpha)
            ax.plot([0, 1000], [0, 1000], c='C0', lw=1)
            ax.legend().set(visible=False)
            ax.set(xscale='log', yscale='log')
            ax.set(xlim=[2000, 1], ylim=[2000, 1])

        if title != '':
            for ax in axs:
                if bit is not None:
                    title1 = f'{title}, n={len(bit)} samples'
                else:
                    title1 = title
                ax.set(title=title1)

    def weight_image(self, ax, levels=20, colorbar=True):
        bit = self.weight_df.weight.unstack()
        img = ax.contourf(bit.columns, bit.index, bit, cmap='viridis_r', levels=levels)
        ax.set(xlabel='p1', ylabel='p0', title='Weight for p1', aspect='equal')
        if colorbar:
            ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16, label='Weight to p_upper')

    def quick_price(self, distortion, a):
        """
        price total to assets a using distortion

        requires distribution_spec has a density_df dataframe with a p_total or p_total

        TODO: add ability to price other lines
        :param distortion:
        :param a:
        :return:
        """

        assert isinstance(self.distribution_spec, (agg.Portfolio, agg.Aggregate))

        df = self.distribution_spec.density_df
        temp = distortion.g(df.p_total.shift(-1, fill_value=0)[::-1].cumsum())[::-1]

        if isinstance(temp, np.ndarray):
            # not aall g functions return Series (you can't guarantee it is called on something with an index)
            temp = pd.Series(temp, index=df.index)

        temp = temp.shift(1, fill_value=0).cumsum() * self.distribution_spec.bs
        return temp.loc[a]

    def principal_extreme_distortion_analysis(self, gs, pricing=False):
        """
        Find the principal extreme distortion analysis to solve for gs = g(s), s=self.cloud_df.index

        Assumes that tvar_cloud has been called and that cloud_df exists
        len(gs) = len(cloud_df)

        E.g., call

            b = Bounds(port)
            b.t_mode = 'u'
            # set premium and asset level a
            b.tvar_cloud('total', premium, a)
            # make gs
            b.principal_extreme_distortion_analysis(gs)

        :param gs: either g(s) evaluated on s = cloud_df.index or the name of a calibrated distortion in
        distribution_spec.dists (created by a call to calibrate_distortions)
        :param pricing: if try, try just using pricing distortions
        :return:
        """

        assert self.cloud_df is not None

        if type(gs) == str:
            s = np.array(self.cloud_df.index)
            gs = self.distribution_spec.dists[gs].g(s)

        assert len(gs) == len(self.cloud_df)

        if pricing:
            _ = self.cloud_df.xs(0, axis=1, level=0, drop_level=False)
            X = _.to_numpy()
            idx = _.columns
        else:
            _ = self.cloud_df
            X = _.to_numpy()
            idx = _.columns
        n = X.shape[1]

        print(X.shape, self.cloud_df.shape)

        # Moore Penrose solution
        mp = pinv(X) @ gs
        logger.info('Moore-Penrose solved...')

        # optimization solutions
        A = np.hstack((X, np.eye(X.shape[0])))
        b_eq = gs
        c = np.hstack((np.zeros(X.shape[1]), np.ones_like(b_eq)))

        lprs = linprog(c, A_eq=A, b_eq=b_eq, method='revised simplex')
        logger.info(
            f'Revised simpled solved...\nSum of added variables={np.sum(lprs.x[n:])} (should be zero for exact)')
        self.lprs = lprs

        lpip = linprog(c, A_eq=A, b_eq=b_eq, method='interior-point')
        logger.info(f'Interior point solved...\nSum of added variables={np.sum(lpip.x[n:])}')
        self.lpip = lpip

        print(lprs.x, lpip.x)

        # consolidate answers
        self.pedw_df = pd.DataFrame({'w_mp': mp, 'w_rs': lprs.x[:n], 'w_ip': lpip.x[:n]}, index=idx)
        self.pedw_df['w_upper'] = self.weight_df.weight

        # diagnostics
        for c in self.pedw_df.columns[:-1]:
            answer = self.pedw_df[c].values
            ganswer = answer[answer > 1e-16]
            logger.info(f'Method {c}\tMinimum parameter {np.min(answer)}\tNumber non-zero {len(ganswer)}')

        return gs

    def ped_distortion(self, n, solver='rs'):
        """
        make the approximating distortion from the first n Principal Extreme Distortions (PED)s using rs or ip solutions

        :param n:
        :return:
        """
        assert solver in ['rs', 'ip']

        # the weight column for solver
        c = f'w_{solver}'
        # pull off the tvar and PED weights
        df = self.pedw_df.sort_values(c, ascending=False)
        bit = df.loc[df.index[:n], [c, 'w_upper']]
        # re-weight partial (method / lp-solve) weights to 1
        bit[c] /= bit[c].sum()
        # multiply lp-solve weights with the weigh_df extreme distortion p_lower/p_upper weights
        bit['c_lower'] = (1 - bit.w_upper) * bit[c]
        bit['c_upper'] = bit.w_upper * bit[c]
        # gather into data frame of p and total weight (labeled c)
        bit2 = bit.reset_index().drop([c, 'w_upper'], 1)
        bit2.columns = bit2.columns.str.split('_', expand=True)
        bit2 = bit2.stack(1).groupby('p')['c'].sum()
        # bit2 has index = probability points and values = weights for the wtd tvar distortion
        d = agg.Distortion.wtd_tvar(bit2.index, bit2.values, f'PED({solver}, {n})')
        return d


def similar_risks_graphs_sa(axd, bounds, port, pnew, roe, prem):
    """
    stand-alone
    ONLY WORKS FOR BOUNDED PORTFOLIOS (use for beta mixture examples)
    Updated version in CaseStudy
    axd from mosaic
    bounds = Bounds class from port (calibrated to some base)it
    pnew = new portfolio
    input new beta(a,b) portfolio, using existing bounds object

    sample: see similar_risks_sample()

    Provenance : from make_port in Examples_2022_post_publish
    """

    df = bounds.weight_df.copy()
    df['test'] = df['t_upper'] * df.weight + df.t_lower * (1 - df.weight)

    # HERE IS ISSUE - should really use tvar with bounds and incorporate the bound
    tvar1 = {p: float(pnew.tvar(p)) for p in bounds.tps}
    df['t1_lower'] = [tvar1[p] for p in df.index.get_level_values(0)]
    df['t1_upper'] = [tvar1[p] for p in df.index.get_level_values(1)]
    df['t1'] = df.t1_upper * df.weight + df.t1_lower * (1 - df.weight)

    roe_d = agg.Distortion('roe', roe)
    tvar_d = agg.Distortion('tvar', bounds.p_star('total', prem))
    idx = df.index.get_locs(df.idxmax()['t1'])[0]
    pl, pu, tl, tu, w = df.reset_index().iloc[idx, :-4]
    max_d = agg.Distortion('wtdtvar', w, df=[pl, pu])

    tmax = float(df.iloc[idx]['t1'])
    print('Ties for max: ', len(df.query('t1 == @tmax')))
    print('Near ties for max: ', len(df.query('t1 >= @tmax - 1e-4')))

    idn = df.index.get_locs(df.idxmin()['t1'])[0]
    pln, pun, tl, tu, wn = df.reset_index().iloc[idn, :-4]
    min_d = agg.Distortion('wtdtvar', wn, df=[pln, pun])

    ax = axd['A']
    plot_max_min(bounds, ax)
    n = len(ax.lines)
    roe_d.plot(ax=ax, both=False)
    tvar_d.plot(ax=ax, both=False)
    max_d.plot(ax=ax, both=False)
    min_d.plot(ax=ax, both=False)

    ax.lines[n + 0].set(label='roe')
    ax.lines[n + 2].set(color='green', label='tvar')
    ax.lines[n + 4].set(color='red', label='max')
    ax.lines[n + 6].set(color='purple', label='min')
    ax.legend(loc='upper left')

    ax.set(title=f'Max ({pl}, {pu}), min ({pln}, {pun})')

    ax = axd['B']
    bounds.weight_image(ax)

    bit = df['t1'].unstack(1)
    ax = axd['C']
    img = ax.contourf(bit.columns, bit.index, bit, cmap='viridis_r', levels=20)
    ax.set(xlabel='p1', ylabel='p0', title='Pricing on New Risk', aspect='equal')
    ax.get_figure().colorbar(img, ax=ax, shrink=.5, aspect=16, label='rho(X_new)')
    ax.plot(pu, pl, '.', c='w')
    ax.plot(pun, pln, 's', ms=3, c='white')

    ax = axd['D']
    plot_lee(port, ax, 'k', lw=1)
    plot_lee(pnew, ax, 'r')

    ax = axd['E']
    pnew.density_df.p_total.plot(ax=ax)
    ax.set(xlim=[-0.05, 1.05], title='Density')

    ax = axd['F']
    plot_max_min(bounds, ax)
    for c, dd in zip(['r', 'g', 'b'], ['ph', 'wang', 'dual']):
        port.dists[dd].plot(ax=ax, both=False, lw=1)
        ax.lines[n].set(c=c, label=dd)
        n += 2
    ax.legend(loc='lower right')

    return df


def similar_risks_example():
    """
    Interesting beta risks and how to use similar_risks_sa


    @return:
    """
    # stand alone hlep from the code; split at program = to run different options
    uw = agg.Underwriter()
    p_base = uw('''
    port UNIF
        agg ONE 1 claim sev 1 * beta 1 1 fixed
    ''')
    p_base.update(11, 1 / 1024, remove_fuzz=True)
    prem = p_base.tvar(0.2, 'interp')
    a = 1
    d = (prem - p_base.ex) / (a - p_base.ex)
    v = 1 - d
    roe = d / v
    prem, roe
    p_base.calibrate_distortions(As=[1], ROEs=[roe], strict='ordered')
    bounds = Bounds(p_base)
    bounds.tvar_cloud('total', prem, a, 128 * 2, 64 * 2, 'interp')
    p_star = bounds.p_star('total', prem, kind='interp')

    smfig = FigureManager(cycle='c', color_mode='color', font_size=10, legend_font='small',
                          default_figsize=(5, 3.5))

    f, axs = smfig(1, 3, (18.0, 6.0), )
    ax0, ax1, ax2 = axs.flat
    axi = iter(axs.flat)
    # all with base portfolio

    bounds.cloud_view(axs.flatten(), 0, alpha=1, pricing=True,
                      title=f'Premium={prem:,.1f}, a={a:,.0f}, p*={p_star:.3f}',
                      distortions=[{k: p_base.dists[k] for k in ['roe', 'tvar']},
                                   {k: p_base.dists[k] for k in ['ph', 'wang', 'dual']}])
    for ax in axs.flatten()[1:]:
        ax.legend(ncol=1, loc='lower right')
    for ax in axs.flatten():
        ax.set(title=None)

    program = '''
    port BETA
        agg TWO 1 claim sev 1 * beta [200 300 400 500 600 7] [600 500 400 300 200 1] wts=6 fixed
        # never worked
        # agg TWO 1 claim sev 1 * beta [1 2000 4000 6000 50] [100 6000 4000 2000 1] wts[0.1875 0.1875 0.1875 0.1875 .25] fixed
        # interior solution:
        # agg TWO 1 claim sev 1 * beta [300 400 500 600 35] [500 400 300 200 5] wts[.125 .25 .125 .25 .25] fixed
        #
        # agg TWO 1 claim sev 1 * beta [50 30 1] [1 40 10] wts=3 fixed
        # agg TWO 1 claim sev 1 * beta [50 30 1] [1 40 10] wts[.375 .375 .25] fixed
    
    '''
    p_new = uw(program)
    p_new.update(11, 1 / 1024, remove_fuzz=True)

    p_new.plot(figsize=(6, 4))

    axd = plt.figure(constrained_layout=True, figsize=(16, 8)).subplot_mosaic(
        '''
        AAAABBFF
        AAAACCFF
        AAAADDEE
        AAAADDEE
    '''
    )
    df = similar_risks_graphs_sa(axd, bounds, p_base, p_new, roe, prem)
    return df


def stand_alone_pricing_work(self, dist, p, kind, roe, S_calc='cumsum'):
    """
    Apply dist to the individual lines of self, with capital standard determined by a, p, kind=VaR, TVaR, etc.
    Return usual data frame with L LR M P PQ  Q ROE, and a

    Dist can be a distortion, traditional, or defaut pricing modes. For latter two you have to input an ROE. ROE
    not required for a distortion.

    :param self: a portfolio object
    :param dist: "traditional", "default", or a distortion (already calibrated)
    :param p: probability level for assets
    :param kind: var (or lower, upper), tvar or epd (note, p should be small for EPD, to pander, if p is large we use 1-p)
    :param roe: for traditional methods input roe

    :return: exhibit is copied and augmented with the stand-alone statistics

    from common_scripts.py
    """
    assert S_calc in ('S', 'cumsum')

    var_dict = self.var_dict(p, kind=kind, total='total', snap=True)
    exhibit = pd.DataFrame(0.0, index=['L', 'LR', 'M', 'P', "PQ", 'Q', 'ROE'], columns=['sop'])

    def tidy_and_write(exhibit, ax, exa, prem):
        """ finish up calculation and store answer """
        roe_ = (prem - exa) / (ax - prem)
        exhibit.loc[['L', 'LR', 'M', 'P', 'PQ', 'Q', 'ROE'], l] = \
            (exa, exa / prem, prem - exa, prem, prem / (ax - prem), ax - prem, roe_)

    if dist == 'traditional - no default':
        # traditional roe method, no allowance for default
        method = dist
        d = roe / (1 + roe)
        v = 1 - d
        for l in self.line_names_ex:
            ax = var_dict[l]
            # no allowance for default
            exa = self.audit_df.at[l, 'EmpMean']
            prem = v * exa + d * ax
            tidy_and_write(exhibit, ax, exa, prem)
    elif dist == 'traditional':
        # traditional but allowing for default
        method = dist
        d = roe / (1 + roe)
        v = 1 - d
        for l in self.line_names_ex:
            ax = var_dict[l]
            exa = self.density_df.loc[ax, f'lev_{l}']
            prem = v * exa + d * ax
            tidy_and_write(exhibit, ax, exa, prem)
    else:
        # distortion method
        method = f'sa {str(dist)}'
        for l, ag in zip(self.line_names_ex, self.agg_list + [None]):
            # use built in apply distortion
            if ag is None:
                # total
                if S_calc == 'S':
                    S = self.density_df.S
                else:
                    # revised
                    S = (1 - self.density_df['p_total'].cumsum())
                # some dist return np others don't this converts to numpy...
                gS = pd.Series(dist.g(S), index=S.index)
                exag = gS.shift(1, fill_value=0).cumsum() * self.bs
            else:
                ag.apply_distortion(dist)
                exag = ag.density_df.exag
            ax = var_dict[l]
            exa = self.density_df.loc[ax, f'lev_{l}']
            prem = exag.loc[ax]
            tidy_and_write(exhibit, ax, exa, prem)

    exhibit.loc['a'] = exhibit.loc['P'] + exhibit.loc['Q']
    exhibit['sop'] = exhibit.filter(regex='[A-Z]').sum(axis=1)
    exhibit.loc['LR', 'sop'] = exhibit.loc['L', 'sop'] / exhibit.loc['P', 'sop']
    exhibit.loc['ROE', 'sop'] = exhibit.loc['M', 'sop'] / (exhibit.loc['a', 'sop'] - exhibit.loc['P', 'sop'])
    exhibit.loc['PQ', 'sop'] = exhibit.loc['P', 'sop'] / exhibit.loc['Q', 'sop']

    exhibit['method'] = method
    exhibit = exhibit.reset_index()
    exhibit = exhibit.set_index(['method', 'index'])
    exhibit.index.names = ['method', 'stat']
    exhibit.columns.name = 'line'
    exhibit = exhibit.sort_index(axis=1)

    return exhibit


def stand_alone_pricing(self, dist, p=0, kind='var', S_calc='cumsum'):
    """

    Run distortion pricing, use it to determine and ROE and then compute traditional and default
    pricing, then consolidate the answer

    :param self:
    :param roe:
    :param p:
    :param kind:
    :return:

    from common_scripts.py
    """
    assert isinstance(dist, (agg.Distortion, agg.spectral.Distortion, list))
    if type(dist) != list:
        dist = [dist]
    ex1s = []
    for d in dist:
        ex1s.append(stand_alone_pricing_work(self, d, p=p, kind=kind, roe=0, S_calc=S_calc))
        if len(ex1s) == 1:
            roe = ex1s[0].at[(f'sa {str(d)}', 'ROE'), 'total']
    ex2 = stand_alone_pricing_work(self, 'traditional - no default', p=p, kind=kind, roe=roe, S_calc=S_calc)
    ex3 = stand_alone_pricing_work(self, 'traditional', p=p, kind=kind, roe=roe, S_calc=S_calc)

    return pd.concat(ex1s + [ex2, ex3])


# class GreatFormatter(ticker.ScalarFormatter):
#     """
#     From Great
#
#     """
#
#     def __init__(self, sci=True, power_range=(-3, 3), offset=True, mathText=True):
#         super().__init__(useOffset=offset, useMathText=mathText)
#         self.set_powerlimits(power_range)
#         self.set_scientific(sci)
#
#     def _set_order_of_magnitude(self):
#         super()._set_order_of_magnitude()
#         self.orderOfMagnitude = int(3 * np.floor(self.orderOfMagnitude / 3))


class FigureManager():
    def __init__(self, cycle='c', lw=1.5, color_mode='mono', k=0.8, font_size=12,
                 legend_font='small', default_figsize=(5, 3.5)):
        """
        Another figure/plotter manager: manages cycles for color/black and white
        from Great utils.py, edited and stripped down
        combined with lessons from MetaReddit on matplotlib options for fonts, background
        colors etc.

        Font size was 9 and legend was x-small

        Create figure with common defaults

        cycle = cws
            c - cycle colors
            w - cycle widths
            s - cycle styles
            o - styles x colors, implies csw and w=single number (produces 8 series)

        lw = default line width or [lws] of length 4

        smaller k overall darker lines; colors are equally spaced between 0 and k
        k=0.8 is a reasonable range for four colors (0, k/3, 2k/3, k)

        https://matplotlib.org/3.1.1/tutorials/intermediate/color_cycle.html

        https://matplotlib.org/3.1.1/users/dflt_style_changes.html#colors-in-default-property-cycle

        https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html

        https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        """

        assert len(cycle) > 0

        # this sets a much smaller base fontsize
        # plt.rcParams.update({'axes.titlesize': 'large'})
        # plt.rcParams.update({'axes.labelsize': 'small'})
        # list(map(plt.rcParams.get, ('axes.titlesize', 'font.size')))
        # everything scales off font size
        plt.rcParams['font.size'] = font_size
        # mpl default is medium
        plt.rcParams['legend.fontsize'] = legend_font
        # see https://matplotlib.org/stable/gallery/color/named_colors.html
        self.plot_face_color = 'lightsteelblue'
        self.figure_bg_color = 'aliceblue'
        # graphics set up
        plt.rcParams["axes.facecolor"] = self.plot_face_color
        # note plt.rc lets you set multiple related properties at once:
        plt.rc('legend', fc=self.plot_face_color, ec=self.plot_face_color)
        # is equivalent to two calls:
        # plt.rcParams["legend.facecolor"] = self.plot_face_color
        # plt.rcParams["legend.edgecolor"] = self.plot_face_color
        plt.rcParams['figure.facecolor'] = self.figure_bg_color

        self.default_figsize = default_figsize
        self.plot_colormap_name = 'cividis'

        # fonts: add some better fonts as earlier defaults
        mpl.rcParams['font.serif'] = ['STIX Two Text', 'Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif',
                                      'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L',
                                      'Utopia', 'ITC Bookman',
                                      'Bookman', 'Nimbus Roman No9 L', 'Times', 'Palatino', 'Charter', 'serif']
        mpl.rcParams['font.sans-serif'] = ['Nirmala UI', 'Myriad Pro', 'Segoe UI', 'DejaVu Sans', 'Bitstream Vera Sans',
                                           'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid',
                                           'Arial',
                                           'sans-serif']
        mpl.rcParams['font.monospace'] = ['Ubuntu Mono', 'QuickType II Mono', 'Cascadia Mono', 'DejaVu Sans Mono',
                                          'Bitstream Vera Sans Mono', 'Computer Modern Typewriter', 'Andale Mono',
                                          'Nimbus Mono L', 'Courier New',
                                          'Courier', 'Fixed', 'Terminal', 'monospace']
        mpl.rcParams['font.family'] = 'serif'
        # or
        # plt.rc('font', family='serif')
        # much nicer math font, default is dejavusans
        mpl.rcParams['mathtext.fontset'] = 'stixsans'

        if color_mode == 'mono':
            # https://stackoverflow.com/questions/20118258/matplotlib-coloring-line-plots-by-iteration-dependent-gray-scale
            # default_colors = ['black', 'grey', 'darkgrey', 'lightgrey']
            default_colors = [(i * k, i * k, i * k) for i in [0, 1 / 3, 2 / 3, 1]]
            default_ls = ['solid', 'dashed', 'dotted', 'dashdot']

        elif color_mode == 'cmap':
            # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            norm = mpl.colors.Normalize(0, 1, clip=True)
            cmappable = mpl.cm.ScalarMappable(
                norm=norm, cmap=self.plot_colormap_name)
            mapper = cmappable.to_rgba
            default_colors = list(map(mapper, np.linspace(0, 1, 10)))
            default_ls = ['solid', 'dashed',
                          'dotted', 'dashdot', (0, (5, 1))] * 2
        else:
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                              '#7f7f7f', '#bcbd22', '#17becf']
            default_ls = ['solid', 'dashed',
                          'dotted', 'dashdot', (0, (5, 1))] * 2

        props = []
        if 'o' in cycle:
            n = len(default_colors) // 2
            if color_mode == 'mono':
                cc = [i[1] for i in product(default_ls, default_colors[::2])]
            else:
                cc = [i[1] for i in product(default_ls, default_colors[:n])]
            lsc = [i[0] for i in product(default_ls, default_colors[:n])]
            props.append(cycler('color', cc))
            props.append(
                cycler('linewidth', [lw] * (len(default_colors) * len(default_ls) // 2)))
            props.append(cycler('linestyle', lsc))
        else:
            if 'c' in cycle:
                props.append(cycler('color', default_colors))
            else:
                props.append(
                    cycler('color', [default_colors[0]] * len(default_ls)))
            if 'w' in cycle:
                if type(lw) == int:
                    props.append(
                        cycler('linewidth', [lw] * len(default_colors)))
                else:
                    props.append(cycler('linewidth', lw))
            if 's' in cycle:
                props.append(cycler('linestyle', default_ls))

        # combine all cyclers
        cprops = props[0]
        for c in props[1:]:
            cprops += c

        mpl.rcParams['axes.prop_cycle'] = cycler(cprops)

    def make_fig(self, nr=1, nc=1, figsize=None, xfmt='great', yfmt='great',
                 places=None, power_range=(-3, 3), sep='', unit='', sci=True,
                 mathText=False, offset=True, **kwargs):
        """

        make grid of axes
        apply format to xy axes

        xfmt='d' for default axis formatting, n=nice, e=engineering, s=scientific, g=great
        great = engineering with power of three exponents

        """

        if figsize is None:
            figsize = self.default_figsize

        f, axs = plt.subplots(nr, nc, figsize=figsize,
                              constrained_layout=True, squeeze=False, **kwargs)
        for ax in axs.flat:
            if xfmt[0] != 'd':
                FigureManager.easy_formatter(ax, which='x', kind=xfmt, places=places,
                                             power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText,
                                             offset=offset)
            if yfmt[0] != 'default':
                FigureManager.easy_formatter(ax, which='y', kind=yfmt, places=places,
                                             power_range=power_range, sep=sep, unit=unit, sci=sci, mathText=mathText,
                                             offset=offset)

        if nr * nc == 1:
            axs = axs[0, 0]

        self.last_fig = f
        return f, axs

    __call__ = make_fig

    @staticmethod
    def easy_formatter(ax, which, kind, places=None, power_range=(-3, 3), sep='', unit='', sci=True,
                       mathText=False, offset=True):
        """
        set which (x, y, b, both) to kind = sci, eng, nice
        nice = engineering but uses e-3, e-6 etc.
        see docs for ScalarFormatter and EngFormatter


        """

        def make_fmt(kind, places, power_range, sep, unit):
            if kind == 'sci' or kind[0] == 's':
                fm = ticker.ScalarFormatter()
                fm.set_powerlimits(power_range)
                fm.set_scientific(True)
            elif kind == 'eng' or kind[0] == 'e':
                fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
            elif kind == 'great' or kind[0] == 'g':
                fm = GreatFormatter(
                    sci=sci, power_range=power_range, offset=offset, mathText=mathText)
            elif kind == 'nice' or kind[0] == 'n':
                fm = ticker.EngFormatter(unit=unit, places=places, sep=sep)
                fm.ENG_PREFIXES = {
                    i: f'e{i}' if i else '' for i in range(-24, 25, 3)}
            else:
                raise ValueError(f'Passed {kind}, expected sci or eng')
            return fm

        # what to set
        if which == 'b' or which == 'both':
            which = ['xaxis', 'yaxis']
        elif which == 'x':
            which = ['xaxis']
        else:
            which = ['yaxis']

        for w in which:
            fm = make_fmt(kind, places, power_range, sep, unit)
            getattr(ax, w).set_major_formatter(fm)


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


def process_memory(show_process=False):
    # memory usage in GB and process id
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    m, p = mem_info.rss, process.pid
    mu = m // 100000000
    m = m / (1 << 30)
    if show_process:
        logger.log(35, f'Process id = {p}\nMemory usage = {m:.3f}GB: |' + '=' * mu)
    else:
        logger.log(35, f'Memory usage = {m:.3f}GB: |' + '=' * mu)


def run_case_in_background(case_id, logLevel=30):
    """
    TODO: Fix up...no longer so easy...

    Run a case and produce all the exhibits in the background.
    Log the output and stderr info to a file

    :param case_id:
    :param base_file:
    :param logLevel:
    @return:
    """

    if platform()[:5] == 'Linux':
        # python anywhere
        pgm = '/home/Yzaamb/.virtualenvs/smve38b/bin/python'
    else:
        pgm = 'python'
    args = shlex.split(f'{pgm} -m case_studies -c {case_id} -g {logLevel}')
    # sterr = Path(datetime.now().strftime(f'log/{case_id}-%Y-%m-%d@%H-%M-%S-err.txt'))
    stout = Path(datetime.now().strftime(f'log/{case_id}-out.txt'))
    sterr = Path(datetime.now().strftime(f'log/{case_id}-err.txt'))
    f_out = stout.open('w', encoding='utf=8')
    f_err = sterr.open('w', encoding='utf=8')
    p = Popen(args, stdout=f_out, stderr=f_err, text=True, encoding='utf-8')
    return p, stout, sterr



class ManualRenderResults():

    APPNAME = 'Pricing Insurance Risk'

    def __init__(self, case_object):
        """
        Create local HTML page for the results datasets. Relies on pricinginsurancerisk templates.

        TODO link them into the directory to keep consistent with the website?

        """
        self.case_object = case_object
        self.case_id = case_object.case_id
        self.env = Environment(loader=FileSystemLoader(Path.home() / 'S/websites/pricinginsurancerisk/templates'), autoescape=(['html', 'xml']))
        # not surprisingly, this doesn't work...will need to package...
        # self.env = Environment(loader=FileSystemLoader('https://www.pricinginsurancerisk.com/templates'), autoescape=(['html', 'xml']))

    @staticmethod
    def now():
        return 'Created {date:%Y-%m-%d %H:%M:%S.%f}'.format(date=datetime.now()).rstrip('0')

    @staticmethod
    def format_name(n):
        """
        case_flags
        """
        ns = n.split('_')
        case_study_names = {'cnc': 'Cat/Noncat', 'hs': "Hurricane/Severe Convective Storm", 'Discrete': "Simple Discrete Example", 'tame': "Tame"}
        n = case_study_names.get(ns[0], ns[0].title())
        other = {'roe': '(ROE fixed)', 'equal': "(10 in two ways)", 'ccocblend': ' CCoC blend calibration', 'extendblend': 'Extend blend calibration' }
        ans = [n]
        for i in ns[1:]:
            ans.append(other.get(i, i.title()))
        return ' '.join(ans)

    @staticmethod
    def get_menu_items(base_dir0, page):
        p = base_dir0.glob('*')
        pp = [(i, i.stat().st_mtime) for i in p]
        links = []
        for i, (pth, _) in enumerate(sorted(pp, key=lambda x: x[1], reverse=True)):
            links.append(f'<a {"" if i < 3 else "class=dropdown-item"} '
                         f'class ="text-white" href="/{page}?case={pth.name}" > {ManualRenderResults.format_name(pth.name)} < / a > \n')
        return links

    def render_exhibits_work(self):
        page = 'results'

        case_description = self.case_object.case_description

        base_dir0 = Path.home() / 'aggregate/cases'
        base_dir = base_dir0 / self.case_id

        if base_dir.exists() is False:
            raise ValueError('{self.case_id} directory not found')

        args = {}
        exhibits = list('ABCDEFGHIJKLMNOPQRSUVWXY')
        exhibits.extend(['T_gross', 'T_net'])
        for t in exhibits:
            p = base_dir / f'{t}.html'
            if p.exists():
                args[t] = p.read_text(encoding='utf-8')
            else:
                args[t] = f'</p>Placeholder for Exhibit {t}.</p>'

        # menu bar items
        links = self.get_menu_items(base_dir0, page)
        template = self.env.get_template('results.html')
        return template.render(title=self.APPNAME,
                               case_name=self.format_name(self.case_id),
                               case_id=self.case_id,
                               case_description=case_description,
                               og_meta={},
                               page=page,
                               subpage='results',
                               links=links,
                               **args,
                               timestamp=self.now(),
                               request_method='MANUAL',
                               requestor_id='ALSO MANUAL')

    def render_extended_work(self):
        page = 'extended'

        base_dir0 = Path.home() / 'aggregate/cases'
        base_dir = base_dir0 / self.case_id
        if base_dir.exists() is False:
            raise ValueError('{self.case_id} directory not found')

        # this is the actual content
        blobs = []
        for p in sorted(base_dir.glob('Z*.html')):
            blobs.append(p.read_text(encoding='utf-8'))

        # menu bar items
        links = self.get_menu_items(base_dir0, page)
        template = self.env.get_template('results_extended.html')
        return template.render(title=self.APPNAME,
                               case_name=self.format_name(self.case_id),
                               case_id=self.case_id,
                               og_meta={},
                               page=page,
                               subpage='extended',
                               links=links,
                               blobs=blobs,
                               timestamp=self.now(),
                               request_method='MANUAL',
                               requestor_id='ALSO MANUAL')

    def rebase(self, h):
        h = h.replace('/static', 'static')
        h = h.replace('static\\cases\\', '')
        h = h.replace(f'src="/{self.case_id}', f'src="{self.case_id}')
        return h

    def render(self):
        h = self.render_exhibits_work()
        # rebase directories
        h = self.rebase(h)
        p = Path.home() / f'aggregate/cases/{self.case_id}_book.html'
        p.write_text(h, encoding='utf-8')

        h1 = self.render_extended_work()
        h1 = self.rebase(h1)
        p1 = Path.home() / f'aggregate/cases/{self.case_id}_extended.html'
        p1.write_text(h1, encoding='utf-8')

        return p, p1


# command line related
def setup_parser():
    """
    Set up all command line options and return parser

    :return:  parser object
    """
    parser = argparse.ArgumentParser(
        description='PIR - Case Study Generator. Usual options -rb to run in color mode with ROE and blend fixes and use extend calibration.',
        epilog='start /d . python -m case_studies -o cnc -rb  will run the cnc case in a background window. Add /min to minimize the window. '
               'start "NAME"... uses NAME as the title for the new window.'
    )
    # Debug group

    # EQUAL = 1 << 0  # discrete in equal model (ch 15); discrete only
    # BOOK = 1 << 1  # book monochrome mode graphs, black lines; alt uses colored lines
    # ROE_FIX = 1 << 2
    # BLEND_FIX = 1 << 3
    # COLOR = 1 << 4  # colored background to graphs
    # CCOC_BLEND = 1 << 5  # use CCoC method to calibrate blend (default is to extend)

    # meta download
    parser.add_argument('-c', '--case', action='store', type=str, default='',
                        help='Selected case id from case_spec.csv database. Use -c all to run all cases.')
    parser.add_argument('-g', '--logger', action="store", type=int, default=30,
                        help='Logger level.')
    parser.add_argument('-l', '--list', action="store_true",
                        help='List available cases and exit.')
    # parser.add_argument('-k', '--book', action="store_true",
    #                     help='Book, monochrome graphs.')
    return parser


if __name__ == '__main__':
    # start "name" /d . python -m case_studies -c tame -l 31

    # for debugging

    parser = setup_parser()
    args = parser.parse_args()

    logger.setLevel(args.logger)

    if args.list:
        print(CaseStudy.case_list())
        exit()

    if args.case == 'all':
        case_ids = CaseStudy.case_list().index
    else:
        case_ids = args.case.split(' ')

    process_memory(True)
    for case_id in case_ids:
        logger.log(35, f'{case_id} creating CaseStudy object')
        case = CaseStudy.factory(case_id)
        process_memory()
        case.full_monty()
