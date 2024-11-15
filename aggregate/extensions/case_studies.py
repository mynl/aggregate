# History
# Removed hacks aimed at reproducing book exhibits (color, roe hack, blend fix etc.) these are
# now all applied by default. Set up for new syntax case specification.
# gridoff removed and grid lines cut out.
# Integrates code from common_header, common_scripts, and hack

import aggregate as agg
from aggregate import Aggregate, round_bucket, make_mosaic_figure, FigureManager, Bounds, plot_max_min
import argparse
from collections import OrderedDict
from datetime import datetime
from inspect import signature
from io import StringIO
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
import os
import pandas as pd
from pandas.io.formats.format import EngFormatter
from pathlib import Path
from PIL import Image
from platform import platform
import psutil
import re
from scipy.integrate import cumulative_trapezoid as cumtrapz
import shlex
from subprocess import Popen
# from titlecase import titlecase as title
import webbrowser

from IPython.display import HTML, display

from .. constants import *

# general set up
pd.set_option("display.float_format", EngFormatter(3, True))
pd.set_option('display.max_rows', 500)

# get the logger
logger = logging.getLogger('aggregate.case_studies')

# up to the user to deal with warnings
# logging.captureWarnings(True)  -> all warnings emitted by the warnings module
# will automatically be logged at level WARNING
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning)
# warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

# utilities
stat_renamer = {'L': 'Expected Loss', 'P': "Premium", 'LR': 'Loss Ratio',
                'M': "Margin", 'PQ': 'Leverage', 'Q': 'Capital', 'ROE': "ROE", 'a': 'Assets',
                'rp': "Return Period", 'epdr': 'EPD Ratio', 'EPDR': 'EPD Ratio'}


def add_defaults(dict_in, kind='agg'):
    """
    add default values to dict_inin. Leave existing values unchanged
    Used to output to a data frame, where you want all columns completed

    :param dict_in:
    :param kind:
    :return:
    """

    print('running add_defaults\n' * 10)

    # use inspect to get the defaults
    # obtain signature
    sig = signature(Aggregate.__init__)

    # self and name --> bound signature
    bs = sig.bind(None, '')
    bs.apply_defaults()
    # remove self
    bs.arguments.pop('self')
    defaults = bs.arguments

    if kind == 'agg':
        defaults.update(dict_in)

    elif kind == 'sev':
        for k, v in defaults.items():
            if k[0:3] == 'sev' and k not in dict_in and k != 'sev_wt':
                dict_in[k] = v


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
         "Dual Moment": "Dual",
         "Blend": "Blend"}.get(x, x)
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
    x = x.title()
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
    _dist_ = ['EL', 'Dist ccoc', 'Dist ph', 'Dist wang', 'Dist dual', 'Dist tvar', 'Dist blend', ]
    _classical_ = ['EL', 'ScaledEPD', 'ScaledTVaR', 'ScaledVaR', 'EqRiskEPD', 'EqRiskTVaR', 'EqRiskVaR',
                   'coTVaR', 'covar']
    _gloss_ = {
        'A': 'PIR Chapter  2, Tables  2.3, 2.5, 2.6, 2.7,     Estimated mean, CV, skewness and kurtosis by line and in  total, gross and net. ',
        'B': 'PIR Chapter  2, Figures 2.2, 2.4, 2.6,          Gross and net densities on a linear and log scale.                              ',
        'C': 'PIR Chapter  2, Figures 2.3, 2.5, 2.7,          Bivariate densities: gross and net with gross sample.                           ',
        'D': 'PIR Chapter  4, Figures 4.9, 4.10, 4.11, 4.12,  TVaR, and VaR for unlimited and limited variables, gross and net.               ',
        'E': 'PIR Chapter  4, Tables  4.6, 4.7, 4.8,          Estimated VaR, TVaR, and EPD by line and in total, gross, and net.              ',
        'F': 'PIR Chapter  7, Table  7.2,                     Pricing summary.                                                                ',
        'G': 'PIR Chapter  7, Table  7.3,                     Details of reinsurance.                                                         ',
        'H': 'PIR Chapter  9, Tables  9.2, 9.5, 9.8,          Classical pricing by method.                                                    ',
        'I': 'PIR Chapter  9, Tables  9.3, 9.6, 9.9,          Sum of parts (SoP) stand-alone vs. diversified classical pricing by method.',
        'J': 'PIR Chapter  9, Tables  9.4, 9.7, 9.10,         Implied loss ratios from classical pricing by method.                           ',
        'K': 'PIR Chapter  9, Table  9.11,                    Comparison of stand-alone and sum of parts premium.',
        'L': 'PIR Chapter  9, Tables  9.12, 9.13, 9.14,       Constant CoC pricing by unit for Case Study.                                    ',
        'M': 'PIR Chapter 11, Figures 11.2, 11.3, 11.4,11.5,  Distortion envelope for Case Study, gross.                                      ',
        'N': 'PIR Chapter 11, Table  11.5,                    Parameters for the six SRMs and associated distortions.                         ',
        'O': 'PIR Chapter 11, Figures 11.6, 11.7, 11.8,       Variation in insurance statistics for six distortions  as $s$ varies.           ',
        'P': 'PIR Chapter 11, Figures 11.9, 11.10, 11.11,     Variation in insurance statistics as the asset limit is varied.                 ',
        'Q': 'PIR Chapter 11, Tables  11.7, 11.8, 11.9,       Pricing by unit and distortion for Case Study.                                  ',
        'R': 'PIR Chapter 13, Table  13.1,                    Comparison of gross expected losses by Case, catastrophe-prone lines.           ',
        'S': 'PIR Chapter 13, Tables  13.2, 13.3, 13.4,       Constant 0.10 ROE pricing for Case Study, classical PCP methods.                ',
        'T': 'PIR Chapter 15, Figures 15.2 - 15.7 (G/N),      Twelve plot.                                                                    ',
        'U': 'PIR Chapter 15, Figures 15.8, 15.9, 15.10,      Capital density by layer.                                                       ',
        'V': 'PIR Chapter 15, Tables  15.35, 15.36, 15.37,    Constant 0.10 ROE pricing for Cat/Non-Cat Case Study, distortion, SRM methods.  ',
        'W': 'PIR Chapter 15, Figure 15.11,                   Loss and loss spectrums.                                                        ',
        'X': 'PIR Chapter 15, Figures 15.12, 15.13, 15.14,    Percentile layer of capital  allocations by asset level.                        ',
        'Y': 'PIR Chapter 15, Tables  15.38, 15.39, 15.40,    Percentile layer of capital  allocations compared to distortion allocations.    ',
    }
    def __init__(self):
        """
        Create an empty CaseStudy.

        Use ``factory`` to populate.

        """
        # mode: html (default) or markdown
        self.mode = 'html'

        # variables set in other functions
        self.gs_values = None
        self.s_values = None
        self.blend_distortions = None
        self.blend_d = None
        self.roe_d = None
        self.dist_dict = None
        self.uw = agg.Underwriter(update=False)

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
        self.cap_table = None
        self.cap_table_total = None
        self.cap_table_marginal = None
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
        self.bs = 0.
        self.log2 = 0
        self.padding = 1

        # graphics defaults
        self.fw = FIG_W
        self.fh = FIG_H
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

    def to_json(self):
        """
        Persist to json in file aggregate/cases/case_id.json

        :param fn:
        :return:
        """
        json.dump(self.to_dict(), (Path.home() / f'aggregate/cases/{self.case_id}.json').
                  open('w', encoding='utf-8'))

    def to_dict(self):
        """
        Definition to dictionary
        :return:
        """
        ans = {}
        # use inspect to get the arguments
        sig = signature(self.factory)
        for k in sig.parameters.keys():
            ob = getattr(self, k)
            # these are not json serializable?!!
            # TODO is this the best?
            if isinstance(ob, np.ndarray):
                ob = list(ob)
            ans[k] = ob
        return ans

    def read_json(self, fn):
        """
        Load from json object.
        Files in aggregate/cases.

        :param fn:
        :return:
        """
        args = json.load((Path.home() / f'aggregate/cases/{fn}.json').open('r', encoding='utf=8'))
        self.factory(**args)

    @staticmethod
    def list():
        """
        List cases with stored json files.

        :return:
        """
        for p in (Path.home() / 'aggregate/cases').glob('*.json'):
            print(p)

    def factory(self, case_id, case_name, case_description,
                a_distribution, b_distribution_gross, b_distribution_net,
                reg_p, roe, d2tc,
                f_discrete, s_values, gs_values, bs, log2, padding):
        """
        Create CaseStudy from case_id and all arguments in generic format with
        explicit reinsurance. The reinsured unit is always B.

        Once created, run ``self.full_monty()s to create all exhibits and
        ``self.browse_exhibits()` to view them.

        :param case_id: unique id for case, must be acceptable as a filename
        :param case_name: name of case
        :param case_description: description of case
        :param a_distribution: DecL program for unit A
        :param b_distribution_gross: DecL program for unit B gross of reinsurance
        :param b_distribution_net: DecL program for unit B net of reinsurance
        :param reg_p: regulatory capital probability threshold
        :param roe: target cost of capital
        :param d2tc:  debt to total capital limit, for better blend distortion
        :param f_discrete: True if the output is a discrete distribution
        :param s_values: list of s values used to calibrate the blended distributions. They correspond
          to the return period and prices for cat bonds with small s values.
        :param gs_values: list of g(s) values
        :param bs: bin size for discrete distributions
        :param log2: log2 of the number of bins for discrete distributions
        :param padding: for update.
        """

        self.case_id = case_id  # originally option_id
        self.case_name = case_name
        self.case_description = case_description
        # DecL programs
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
        self.blend_distortions = None
        self.s_values = s_values
        self.gs_values = gs_values
        self.log2 = log2
        self.bs = bs
        self.padding = padding

        # new style output from uw as list of Answer(...) object
        out = self.uw.write(f'port Gross_{self.case_id} {self.a_distribution} {self.b_distribution_gross}')
        self.gross = out[0].object
        # sort out better bucket
        if self.bs == 0 or np.isnan(self.bs):
            self.bs = self.gross.best_bucket(self.log2)
        self.gross.update(log2=self.log2, bs=self.bs, padding=self.padding, remove_fuzz=True)

        out = self.uw.write(f'port Net_{self.case_id} {self.a_distribution} {self.b_distribution_net}')
        self.net = out[0].object
        self.net.update(log2=self.log2, bs=self.bs, remove_fuzz=True)
        self.ports = OrderedDict(gross=self.gross, net=self.net)

        self.pricings = OrderedDict()
        self.pricings['gross'] = pricing(self.gross, reg_p, roe)
        self.pricings['net'] = pricing(self.net, reg_p, roe)

        # are these used?
        self.re_type = self.net[self.re_line].reinsurance_kinds()
        self.re_description = self.net[self.re_line].reinsurance_description(kind='both')
        # TODO replace! Output as Table G in Ch 7
        # self.re_summary = pd.DataFrame([self.re_line, self.re_type, self.re_attach_p,
        #                                 self.re_attach, self.re_detach_p,
        #                                 self.re_detach - self.re_attach], index=[
        #     'Reinsured Line', 'Reinsurance Type', 'Attachment Probability', 'Attachment', 'Exhaustion Probability',
        #     'Limit'], columns=[self.case_id])
        # self.re_summary.index.name = 'item'

        # cap table needs pricing_summary
        self.make_audit_exhibits()

        # set up the common distortions
        # make the cap table
        logger.info('Calibrating blend distortions')
        self.make_cap_table()
        prem = self.pricings['gross'].at['P', 'total']
        a = self.pricings['gross'].at['a', 'total']
        self.blend_distortions = self.gross.calibrate_blends(a, prem, s_values, gs_values, debug=False)

        self.roe_d = self.approx_roe(e=1e-10)
        self.dist_dict = OrderedDict(ccoc=self.roe_d, **self.blend_distortions)
        k = list(self.blend_distortions.keys())[0]
        self.blend_d = self.blend_distortions[k]
        self.cache_dir = self.cache_base / f'{self.case_id}'
        self.cache_dir.mkdir(exist_ok=True)

    def __repr__(self):
        return f'''Case Study object {self.case_id} @ {self.cache_dir} ({super().__repr__()})
Portfolios: {self.gross.name} (EL={self.gross.agg_m:.2f}) and {self.net.name} ({self.net.est_m:.2f}).
Lines: {", ".join(self.gross.line_names)} (ELs={", ".join([f"{a.agg_m:.2f}" for a in self.gross])}).
'''
    def browse_exhibits(self):
        # book exhibits
        webbrowser.open(Path.home() / f'aggregate/cases/{self.case_id}_book.html')
        # extended exhibits
        webbrowser.open(Path.home() / f'aggregate/cases/{self.case_id}_extended.html')

    def full_monty(self, render=True):
        """
        All updating and exhibit generation. No output. For use with command line.

        :param self:
        :return:
        """
        assert self.mode in ('html', 'markdown'), f'ERROR: mode must be html or markdown, not {self.mode}'

        logger.info('Start Full Monty Update')
        self.make_all()
        process_memory()

        logger.info('display exhibits')
        self.show = False
        self.show_exhibits('all')
        # process_memory()
        self.show_extended_exhibits()
        process_memory()

        logger.info('create graphics')
        self.show_graphs('all')
        process_memory()
        self.show_extended_graphs()
        process_memory()
        logger.info(f'{self.case_id} computed')

        if render:
            # save the results
            if self.mode == 'html':
                mrr = ManualRenderResults(self)
            elif self.mode == 'markdown':
                mrr = ManualRenderResultsMarkdown(self)
            mrr.render_custom('[A-Y]', suffix='book')
            mrr.render_custom('Z.*', suffix='extended')
            logger.info(f'{self.case_id} saved to {"HTML" if self.mode=="html" else "markdown"}...complete!')

    def render_custom(self, outdir):
        """
        For markdown only, often don't want to render to ~/aggregate/cases
        This method renders to user specified directory, outdir.
        """
        if self.mode != 'markdown':
            raise ValueError('This method is only for markdown mode')
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        mrr.render_custom('[A-Y]', suffix='book')
        mrr.render_custom('Z.*', suffix='extended')
        logger.info(f'{self.case_id} saved to {"HTML" if self.mode=="html" else "markdown"}...complete!')




    def approx_roe(self, e=1e-15):
        """
        Make an approximation to the ccoc distortion with no mass, using a slope from ``e`` to 0.

        :param e:
        :return:
        """
        aroe = pd.DataFrame({'col_x': [0, e, 1], 'col_y': [0, self.v * e + self.d, 1]})
        approx_roe_di = agg.Distortion('convex', None, None, df=aroe, col_x='col_x', col_y='col_y', display_name='ccoc')
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

    @classmethod
    def _display_work(cls, exhibit_id, df, caption, ff=None, save=True,
                      cache_dir=None, show=False, align='right'):
        """
        Allow calling without creating an object, e.g. to consistently format the _cases_ database

        additional_properties is a list of selector, property pairs
        E.g. [(col-list, dict-of-properties)]

        :param exhibit_id:
        :param df:
        :param caption:
        :param ff:
        :param save:
        :return:
        """

        # revised set: used in agg, greys look better
        cell_hover = {
            'selector': 'td:hover',
            'props': [('background-color', '#ffffb3')]
        }
        index_names = {
            'selector': '.index_name',
            'props': 'font-style: italic; color: white; background-color: #777777; '
                     'font-weight:bold; border: 1px solid white; text-transform: capitalize; '
                     'text-align:left;'
        }
        headers = {
            'selector': 'th:not(.index_name)',
            'props': 'background-color: #DDDDDD; color: black;  border: 1px solid #ffffff;'
        }
        center_heading = {
            'selector': 'th.col_heading',
            'props': 'text-align: center;'
        }
        left_index = {
            'selector': '.row_heading',
            'props': 'text-align: left;'
        }
        # note: this is only difference with generic aggregate method:
        td = {
            'selector': 'td',
            'props': f'text-align: {align};'
        }
        all_styles = [cell_hover, index_names, headers, center_heading,  left_index, td]
        # do the styling
        styled_df = df.style.set_table_styles(all_styles)

        if exhibit_id != '':
            caption = f'({exhibit_id}) {caption}'

        styled_df = styled_df.format(ff).set_caption(caption) # .set_table_attributes(f'id={exhibit_id}')

        if show is True: # ? and save is True:
            display(styled_df)
        if save is True:
            styled_df.to_html(buf=Path(cache_dir / f'{exhibit_id}.html'))
        # shut up already...
        # process_memory()
        return styled_df

    @classmethod
    def _display_work_md(cls, exhibit_id, df, caption, ff=None, save=True,
                         cache_dir=None, show=False):
        """
        Allow calling without creating an object, e.g. to consistently format the _cases_ database

        additional_properties is a list of selector, property pairs
        E.g. [(col-list, dict-of-properties)]

        :param exhibit_id:
        :param df:
        :param caption:
        :param ff:
        :param save:
        :return:
        """

        if exhibit_id != '':
            caption = f'({exhibit_id}) {caption}'

        styled_df = df.style.format(ff).set_caption(caption)

        if show is True: # ? and save is True:
            display(styled_df)
        if save is True:
            dfc = df.copy()
            if isinstance(ff, dict):
                for c in dfc.columns:
                    if c in ff.keys():
                        dfc[c] = dfc[c].apply(ff[c])
            elif callable(ff):
                dfc = dfc.applymap(ff)
            else:
                print(f'Unhandled type ff = {type(ff)}')
            if isinstance(dfc.columns, pd.MultiIndex):
                dfc.columns = [': '.join(col).strip() for col in dfc.columns.values]
            if isinstance(dfc.index, pd.MultiIndex):
                dfc.index = [': '.join(col).strip() for col in dfc.index.values]
            txt = dfc.to_markdown(headers='keys',
                                  tablefmt='pipe',
                                  stralign='default'
            )
            if caption != '':
                caption = caption.replace('\n', ' ')
            with Path(cache_dir / f'{exhibit_id}.md').open('wt', encoding='utf-8') as f:
                f.write(f'## Table {exhibit_id}\n\n')
                if exhibit_id in cls._gloss_:
                    f.write(f'{cls._gloss_[exhibit_id]}\n\n')
                f.write(txt)
                f.write(f'\n: {caption}\n')
        # TODO: what is done with the return item?
        # placeholder: make it the same as  html... really only interested in the
        # saving side-effects.
        return styled_df

    def _display(self, exhibit_id, df, caption, ff=None, save=True):
        """

        caption = string caption
        ff = float format function

        """
        # figure the optimal by column formatting
        fc = lambda x: f'{x:,.1f}'
        f3 = lambda x: f'{x:.3f}'
        f5g = lambda x: f'{x:.5g}'
        # fp = lambda x: f'{x:.1%}'
        # guess sensible defaults
        if ff is None:
            fmts = {}
            for n, r in df.agg([np.mean, np.min, np.max, np.std]).T.iterrows():
                if r['max'] < 1 and r['min'] < 1e-4 and r['min'] > 0:
                    fmts[n] = f5g
                elif r['max'] < 1 and r['min'] > 1e-4:
                    fmts[n] = f3
                elif r['max'] < 10 and r['min'] >= 0:
                    fmts[n] = f3
                else:
                    # was fc, but none of these numbers is that large
                    fmts[n] = f3
            ff = fmts
        logger.info(f'Exhibit {exhibit_id} processed')
        if self.mode == 'html':
            return self._display_work(exhibit_id, df, f'<div id="{exhibit_id}" /> ' + caption, ff, save,
                                  self.cache_dir, self.show)
        elif self.mode == 'markdown':
            return self._display_work_md(exhibit_id, df, caption, ff, save,
                                  self.cache_dir, self.show)
        # return self._display_work(exhibit_id, df, caption, ff, save, self.cache_dir, self.show)

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
            self._display("F", urn(self.pricing_summary), caption, lambda x: f'{x:,.3f}')

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
            caption = f"""Comparison of stand-alone and sum of parts (SoP) premium for {self.case_name}. Reductions shown as percentage change."""
            tab911 = self.modern_monoline_sa.loc[
                         [('No Default', 'Loss'), ('No Default', 'Premium'), ('No Default', 'Capital'),
                          ('With Default', 'Loss'), ('With Default', 'Premium'), ('With Default', 'Capital')]].iloc[:,
                     [2, 3, 5, 6]]
            tab911.columns = ['Gross SoP', 'Gross Total', 'Net SoP', 'Net Total']
            tab911['Gross Redn'] = tab911['Gross Total'] / tab911['Gross SoP'] - 1
            tab911['Net Redn'] = tab911['Net Total'] / tab911['Net SoP'] - 1
            tab911 = tab911.iloc[:, [0, 1, 4, 2, 3, 5]]
            self._display("K", tab911, caption, lambda x: f'{x:,.1f}' if x > 10 else f'{x:.1%}'.replace('%', ''))

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
                ['ccoc', 'ph', 'wang', 'dual', 'tvar'], ['param', 'error', '$P$', '$K$', 'ROE', '$S$']]. \
                rename(columns={'$P$': 'P', '$K$': 'K', '$S$': 'S', 'ROE': 'ι'})
            self._display("N", urn(tab115), caption, None)

            # table_no = [11.7, 11.8, 11.9][self.case_number]
            # caption = f'Table {table_no}: '
            if self.show: display(HTML('<hr>'))
            caption = f"""Pricing by unit and distortion for {self.case_name}, calibrated to
CCoC pricing with {self.roe} cost of capital and p={self.reg_p}.
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
        :return:
        """

        if self.f_discrete is True:
            bit = self.make_discrete()
            caption = 'Tables 11.1 and 11.2: computing expected loss and premium, PH 0.5 distortion.'
            self._display("Z-11-1", bit, caption) # , self.default_float_format2)

            bit = self.gross.augmented_df.query('loss==80').filter(regex='^(exag?_total)$').copy()
            bit.index.name = 'a'
            bit.columns = ['Limited Loss', 'Premium']
            caption = 'Table 11.3 outcomes with assets a=80 (answers only), PH 0.5 distortion.'
            self._display("Z-11-2", bit, caption) # , self.default_float_format2)

            bit = self.make_discrete('tvar')
            caption = 'Table 11.4 (Exercise): computing expected loss and premium, calibrated TVaR distortion.'
            self._display("Z-11-3", bit, caption) #, self.default_float_format2)

            bit = self.gross.augmented_df.query('loss==80').filter(regex='^(exag?_total)$').copy()
            bit.index.name = 'a'
            bit.columns = ['Limited Loss', 'Premium']
            caption = 'Table (new) outcomes with assets a=80 (answers only), TVaR distortion.'
            self._display("Z-11-4", bit, caption) # , self.default_float_format2)

        else:
            # progress of benefits - always show
            self.make_progression()
            caption = f'Table (new): Premium stand-alone by unit, sum, and total, and natural allocation by distortion.'
            self._display('Z-15-1', urn(self.progression.xs('Premium', axis=0, level=1)['gross']), caption)

            f1 = lambda x: f'{x:.3f}'
            fp = lambda x: f'{x:.0%}'
            fp1 = lambda x: f'{x:.1%}'
            ff = dict(zip(self.walk.columns, (f1, f1, f1, fp1, fp1, fp1, fp, fp, fp)))
            # TODO be explicit about the distortion
            caption = 'Table (new): Progression of premium benefit by distortion. ' \
                      'Differences between allocated and stand-alone, ' \
                      'the implied premium reduction, and the split by unit.'
            self._display('Z-15-2', self.walk, caption, ff)

            # cap table related
            for kind in ['gross', 'net']:
                self.make_cap_table(kind=kind)
                self.ports[kind].apply_distortion(self.blend_d)
                caption = f'Table (new): {kind.title()} asset tranching with S&P bond default rates and a {self.d2tc:.1%} debt to ' \
                          'total capital limit.'
                self._display(f'Z-TR-{kind[0]}', self.cap_table, caption);
                self._display(f'Z-TRT-{kind[0]}', self.cap_table_total,
                              f'Return by tranche, total {kind} view, blend distortion.');
                self._display(f'Z-TRM-{kind[0]}', self.cap_table_marginal,
                              f'Return by tranche, marginal {kind} view, blend distortion.');
                self.show_tranching_graph(kind)

    def make_cap_table(self, kind='gross'):
        """
        TODO: this should be a Portfolio method. Input should be a. prem and EL are computable given
        distortion has been applied.
        XXXX Don't want that... want to be able to apply without ref to augmented df
        Suggest reasonable debt tranching for kind=(gross|net) subject to self.d2tc debt to total capital limit.
        Uses S&P bond default analysis.

        Creates self.cap_table and self.debt_stats

        This is a cluster. There must be a better way...

        This version did all the difficult tranching...which is based on dubious info anyways...

        """

        port = self.ports[kind]
        a, premium, el = self.pricing_summary.loc[['a', 'P', 'L'], kind]
        capital = a - premium
        debt = self.d2tc * capital
        equity = capital - debt
        debt_attach = a - debt
        prob_debt_attach = port.sf(debt_attach)
        cap_table = port.density_df.loc[[port.snap(i) for i in [a, premium+equity, premium, el]],
                                  ['loss', 'F', 'S']]
        cap_table['Amount'] = -np.diff(cap_table.loss, append=0)
        # add prob attaches
        cap_table['Pct Assets'] = cap_table.Amount / cap_table.iloc[0, 0]
        cap_table['Cumul Pct'] = cap_table.loss / cap_table.iloc[0, 0]
        cap_table['Attaches'] = cap_table.loss.shift(-1, fill_value=0)
        cap_table = cap_table.rename(columns={'loss': 'Exhausts', 'S': 'Pr Attaches'})
        cap_table['Pr Attaches'] = cap_table['Pr Attaches'].shift(-1, fill_value=1)
        cap_table['Pr Exhausts'] = [port.sf(i) for i in cap_table.Exhausts]
        cap_table = cap_table[
            ['Amount', 'Pct Assets', 'Attaches', 'Pr Attaches', 'Exhausts', 'Pr Exhausts', 'Cumul Pct']]
        cap_table.index = ['Debt', 'Equity', 'Premium', 'Loss']
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

            bit.index = self.cap_table_total.index
            bit.loc['Total', :] = bit.sum()
            bit.loc['Total', 'lr'] = bit.loc['Total', 'exa_total'] / bit.loc['Total', 'exag_total']
            bit.loc['Total', 'coc'] = bit.loc['Total', 'margin'] / bit.loc['Total', 'capital']
            bit['Discount'] = bit.coc / (1 + bit.coc)
            bit = bit.rename(columns=marginal_renamer)
            self.cap_table_marginal = bit

        # return ans, tranches, bit, spdf, cap_table
        # self.debt_stats = pd.Series(
        #     [a, el, premium, capital, equity, debt, debt_attach, prob_debt_attach, attach_rating, exhaust_rating],
        #     index=['a', 'EL', 'P', 'Capital', 'Equity', 'Debt', 'D_attach', 'Pr(D_attach)', 'Attach Rating',
        #            'Exhaust Rating'])
        self.cap_table = cap_table

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

    def get_f_axs(self, nr, nc):
        w = nc * self.fw
        h = nr * self.fh
        return self.smfig(nr, nc, (w, h))

    @staticmethod
    def get_image_dimensions(image_path):
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)

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
            dgross = 'ccoc'
        elif self.case_id.startswith('tame'):
            dnet = 'tvar'
            dgross = 'ccoc'
        elif self.case_id.startswith('cnc'):
            dnet = 'ph'
            dgross = 'dual'
            # dnet = 'ccoc'
            # dgross = 'ccoc'
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
        if self.mode == 'html':
            blob = f"""
<figure id="{plot_id}">
<img src="/{pth}" width="100%" alt="Figure {f}" style="width:{100}%">
<figcaption class="caption">{caption}</figcaption>
</figure>
"""
            (self.cache_dir / f'{plot_id}.html').write_text(blob, encoding='utf-8')
        elif self.mode == 'markdown':
            if plot_id in self._gloss_:
                gloss = f'{self._gloss_[plot_id]}\n\n'
            else:
                gloss = ''
            w, h = self.get_image_dimensions(p)
            scale = int(w / 7200 * 75)
            blob = f"""## Figure {plot_id}

{gloss}
            
![{caption}](/{pth}){{width="{scale}%"}}
"""
            (self.cache_dir / f'{plot_id}.md').write_text(blob, encoding='utf-8')
        process_memory()

    def case_twelve_plot(self):
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
            # ad comps does not create augmented
            port.apply_distortion(port.dists['dual'], efficient=False)
            port.twelve_plot(f, axs, xmax=xlim[1], ymax2=xlim[1],
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
        TODO this is over the top and too slow.

        :return:
        """
        if self.f_discrete:
            return
        # blended distortion
        f, axs = self.smfig(1, 3, (12.0, 4.0), )
        ax0, ax1, ax2 = axs.flat

        ps = np.hstack((np.linspace(0, 0.1, 2000, endpoint=False), np.linspace(0.1, 1, 100)))
        ph = self.gross.dists['ph'].g(ps)

        # ax.plot(ps, blend, label='Naive blend')
        lsi = iter([':', '--', '-.'] * 5)
        for k, v in self.dist_dict.items():
            temp = v.g(ps)
            ls = next(lsi)
            for ax in axs.flat:
                ax.plot(ps, temp, lw=1, ls=ls, label=k)
        for ax in axs.flat:
            ax.plot(ps, ps, c='k', lw=0.5)
            ax.plot(self.s_values, self.gs_values, 'x', lw=.5, label='Calibration pricing data')
            ax.plot(ps, ph, lw=0.5, label='PH')
            ax.legend()
        ax1.set(xlim=[0, 0.1], ylim=[0, 0.1])
        ax2.set(xscale='log', yscale='log', xlim=[.8e-6, 1], ylim=[.8e-6, 1])

        caption = f'(Z-BL-1) Figure (new): Calibrated blend distortions, compard to CCoC. ' \
                    'The x marks show default and pricing data used in calibration. ' \
                    'Middle plot zooms into 0 < s < 0.1. Righthand plot uses a log/log scale. ' \
                    'PH distortion added for comparison.'
        self._display_plot('Z-BL-1', f, caption)
        return f

    def show_tranching_graph(self, kind):
        """
        graph of tranching
        revised vertical view
        kind = gross or net
        called by show_extended_exhibits

        """
        port = self.ports[kind]

        # new style = vertical, but focused on what is shown in the cap_table range
        f, axs = make_mosaic_figure('AB', w=4, h=6, return_array=True)
        ax0, ax1 = axs.flat
        for ax in axs.flat:
            if ax is ax0:
                ax.plot(port.density_df.F, port.density_df.loss,
                    lw=1.5, label='Loss')
            else:
                ax.plot(1 / port.density_df.S, port.density_df.loss,
                    lw=1.5, label='Loss')

            col = 0
            for n, x in self.cap_table_total.iterrows():
                col += 1
                ax.axhline(x.Assets, c=f'C{col}', lw=1.5, label=f'{n}, ¤{x.Assets:,.0f} @ p={x.Adequacy:.3%}')
        pts = self.cap_table_total.Assets
        ax.yaxis.set_major_locator(ticker.FixedLocator(pts.values))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter([f'{i}: ¤{v:,.0f}' for i, v in pts.items()]))

        ax0.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax0.axvline(0, lw=0.25, c='k')
        ax0.axvline(1, lw=0.25, c='k')
        ax1.axvline(1 / (1 - self.cap_table_total.Adequacy[0]), lw=0.25, c='k')
        ymin = port.q(0.0001)
        ymax = 1.05 * self.cap_table_total.Assets[0]
        ax0.set(ylabel='Loss', xlabel='Non exceedance probability')
        ax0.set(ylim=[ymin, ymax], yscale='linear')
        ax1.set(ylim=[ymin, ymax], xlim=[0.8, 2 / (1 - self.cap_table_total.Adequacy[0])],
                xscale='log', xlabel='Log return period')
        ax0.legend(loc='upper left')
        caption = f'(Z-TR-1) Figure (new): {kind} capital tranching with a {self.d2tc:.1%} debt ' \
                    'to total capital limit.'
        self._display_plot(f'Z-TR-1-{kind[0]}', f, caption)

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
        roe_d = agg.Distortion('ccoc', self.roe)
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

        ax.lines[n + 0].set(label='ccoc')
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
        :return:
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
            dist_list = ['ccoc', 'ph', 'wang', 'dual', 'tvar', 'blend']
            for dn in dist_list:
                gprime[dn] = diff(self.gross.dists[dn].g)

            self.diff_g = pd.DataFrame({dn: gprime[dn](1 - ps) for dn in dist_list},
                                       index=bit.index)

            names = {'ccoc': 'CCoC', 'ph': 'Prop Hazard', 'wang': 'Wang',
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
                if d1 == 'ccoc':
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
                bit0 = bit0.reset_index(drop=False).groupby('qs').agg(np.mean).set_index('return')
                self.exeqa[port.name] = bit0

            ps = 1 / bit0.index
            gprime = {}
            dist_list = ['ccoc', 'ph', 'wang', 'dual', 'tvar', 'blend']
            for dn in dist_list:
                gprime[dn] = diff(self.gross.dists[dn].g)

            self.diff_g = pd.DataFrame({dn: gprime[dn](ps) for dn in dist_list},
                                       index=bit0.index)

            names = {'ccoc': 'CCoC', 'ph': 'Prop Hazard', 'wang': 'Wang',
                     'dual': "Dual", 'tvar': "TVaR", 'blend': "Blend"}

            lbl = 'Gross E[Xi | X]'
            for (k, b), ax in zip(self.exeqa.items(), axs.flat):
                b.plot(ax=ax)
                ylim = {'tame': 200, 'cnc': 500, 'hs': 2500}.get(self.case_id, self.gross.limits('range', 'log')[1])
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
                if d1 == 'ccoc':
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

        caption = (
            f'(W) Figure 15.11: {self.case_name}, loss spectrum (gross/net top row). Rows 2 and show VaR weights '
            'by distortion. In the second row, the CCoC distortion includes a mass putting weight '
            f'𝑑 = {self.roe}∕{1 + self.roe} at the maximum loss, corresponding to an infinite density. '
            'The lower right-hand plot compares all five distortions on a log-log scale.')
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
                        tvar = bounds.tvar_unlimited_function(pu)
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
            bounds.cloud_view(axs=axs, n_resamples=0, alpha=1, pricing=True,
                              title=f'Premium={premium:,.1f}, p={self.reg_p:.1g}\na={a:,.0f}, p*={p_star:.3f}',
                              distortions=[{k: port.dists[k] for k in ['ccoc', 'tvar']},
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

            for dn, ls in zip(['ccoc', 'tvar', 'blend'], ['-', '--', ':']):
                axi = iter(axs.flatten()[6:])
                g_ins_stats(axi, dn, port.dists[dn], ls=ls)

            caption = f'(O) {self.case_name}, variation in premium, loss ratio, markup (premium to loss), margin, discount rate, and premium to capital leverage for six distortions, shown in two groups of three. Top six plots show proportional hazard, Wang, and dual moment; lower six: CCoC, TVaR, and Blend.'
            self._display_plot('O', f, caption)

        if 11 in chapters or 11.9 in chapters:
            # fig 11.9, 10, 11
            logger.info('Figures 11.9, 11.10, 11.11')
            f, axs = self.smfig(6, 4, (4 * 2.5, 6 * 2.5))

            port = self.ports['gross']
            for i, dn in zip(range(6), ['ccoc', 'ph', 'wang', 'dual', 'tvar', 'blend']):
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
            self.case_twelve_plot()

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
        port.dists['ccoc'] = self.roe_d
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
            bitgross = port.stand_alone_pricing(self.roe_d, p=self.reg_p, kind='var')
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
            if 'ccoc' not in port.dists:
                port.dists['ccoc'] = self.dist_dict['ccoc']
            if 'blend' not in port.dists and 'blend' in self.dist_dict:
                port.dists['blend'] = self.dist_dict['blend']
            elif 'blend' not in port.dists:
                # TODO arbitrary?!
                k = list(self.blend_distortions.keys())[-1]
                port.dists['blend'] = self.dist_dict[k]

        # Ch. Modern Mono Practice across distortions
        # all using GROSS calibrated distortions:
        distortions = [self.ports['gross'].dists[dn] for dn in ['ccoc', 'ph', 'wang', 'dual', 'tvar', 'blend']]
        bit_apply_gross = []
        for port in self.ports.values():
            temp = port.stand_alone_pricing(distortions, p=self.reg_p, kind='var')
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
                                                    regex='ph|dual|wang|tvar|ccoc|blend')
                ad_compss[k] = ad_comps
                if k == 'net':
                    # for classical CCoC need to just use roe distortion
                    ad_compss['net_classical'] = port.analyze_distortions(p=self.reg_p,
                                                                          efficient=False,
                                                                          regex='ccoc')
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
        dns = ['Ccoc', 'PH', 'Wang', 'Dual', 'TVaR', 'Blend']
        idx = product(vas, dns)
        bit_sa = self.modern_monoline_by_distortion.loc[idx]

        vas = ['P', 'LR', 'ROE']
        dns = ['Dist ccoc', 'Dist ph', 'Dist wang', 'Dist dual', 'Dist tvar', 'Dist blend']
        idx = product(vas, dns)
        renamer = {'P': 'Premium', 'LR': 'Loss Rato', 'ROE': 'Rate of return', 'Dist ccoc': 'CCoC', 'Dist ph': 'PH',
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
        logger.info('boundss and pstar done')
        self.make_classic_pricing()
        process_memory()
        logger.info('classic pricing done')
        self.make_modern_monoline()
        process_memory()
        logger.info('modern monoline done')
        self.make_ad_comps()
        process_memory()
        logger.info('ad comps done')
        self.apply_distortions(dnet, dgross)
        process_memory()
        logger.info('apply distortions done')
        self.make_bodoff_comparison()
        process_memory()
        logger.info('Bodoff exhibits done')

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
          'ccoc': 'CCoC',
          'tvar': 'TVaR',
          'blend': 'Blend'}[dn0]
    if dn0 == 'blend':
        dn = 'Blend'
    elif dn0 == 'ccoc':
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
          'ccoc': 'CCoC',
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


def process_memory(show_process=False):
    # memory usage in GB and process id
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    m, p = mem_info.rss, process.pid
    mu = m // 100000000
    m = m / (1 << 30)
    if show_process:
        logger.info(f'Process id = {p}\nMemory usage = {m:.3f}GB: |' + '=' * mu)
    else:
        logger.info(f'Memory usage = {m:.3f}GB: |' + '=' * mu)


class ManualRenderResults():

    APPNAME = 'Pricing Insurance Risk'

    def __init__(self, case_object):
        """
        Create local HTML page for the results datasets. Based on pricinginsurancerisk templates.
        Uses pricinginsurancerisk.com css file.

        """
        self.templates = Path(agg.__file__).parent / 'templates'
        self.case_object = case_object
        self.env = Environment(loader=FileSystemLoader(self.templates),
                               autoescape=(['html', 'xml']))

    @staticmethod
    def now():
        return 'Created {date:%Y-%m-%d %H:%M:%S.%f}'.format(date=datetime.now()).rstrip('0')

    def render(self):
        """
        Render all the exhibits. To render the book exhibits only (without the
        extended exhibits) run .render_custom('[A-Y]')

        :return:
        """
        self.render_custom('[A-Y]', suffix='book')
        self.render_custom('Z.*', suffix='extended')

    def render_custom(self, *argv, suffix='custom'):
        """
        Render a custom list of exhibits using standard glob file expansion.

        The book exhibits are A-Y with Tn and Tg net and gross.

        All extended exhibits are Z...

        :param argv:
        :return:
        """
        base_dir0 = Path.home() / 'aggregate/cases'
        base_dir = base_dir0 / self.case_object.case_id
        if base_dir.exists() is False:
            raise ValueError('{self.case_id} directory not found')

        # this is the actual content
        blobs = []
        ids = []
        for pattern in argv:
            for p in sorted(base_dir.glob('*.html')):
                if re.match(pattern, p.stem):
                    ids.append(p.stem)
                    blobs.append(p.read_text(encoding='utf-8'))

        desc = [f'<p>{self.case_object.case_description}</p>']
        spec = self.case_object.to_dict()
        desc.append(f'<p>Distributions</p><pre>{spec["a_distribution"]}')
        desc.append(f'{spec["b_distribution_gross"]}')
        desc.append(f'{spec["b_distribution_net"]}</pre>')
        desc.append(f'<p>Other parameters: ')
        # desc.append('<ul>')
        desc.append(f'reg_p = {spec["reg_p"]}, ')
        desc.append(f'roe = {spec["roe"]}, ')
        desc.append(f'd2tc = {spec["d2tc"]}, ')
        desc.append(f's_values = {spec["s_values"]}, ')
        desc.append(f'gs_values = {spec["gs_values"]}, ')
        desc.append(f'f_discrete = {spec["f_discrete"]}, ')
        desc.append(f'log2 = {spec["log2"]}, ')
        desc.append(f'bs = {spec["bs"]}, and')
        desc.append(f'padding = {spec["padding"]}.</p>')
        # desc.append('</ul>')
        desc = '\n'.join(desc)

        # menu bar items
        template = self.env.get_template('results_extended.html')
        h = template.render(title=self.APPNAME,
                               case_name=self.case_object.case_name,
                               case_id=self.case_object.case_id,
                               case_description=desc,
                               subpage='extended',
                               blobs=blobs,
                               ids=ids,
                               timestamp=self.now(),
                               templates=self.templates
                            )

        # rebase images, entered in snippets as /{case_id}/...
        # other rebasing issues removed because we use the custom installed template
        h = h.replace(f'src="/{self.case_object.case_id}', f'src="{self.case_object.case_id}')
        p = Path.home() / f'aggregate/cases/{self.case_object.case_id}_{suffix}.html'
        p.write_text(h, encoding='utf-8')
        logger.info(f'Rendered {len(blobs)} exhibits and plots.')


class ManualRenderResultsMarkdown():
    def __init__(self, case_object):
        """
        Create local markdown page for the results datasets.

        """
        self.case_object = case_object

    @staticmethod
    def now():
        return 'Created {date:%Y-%m-%d %H:%M:%S.%f}'.format(date=datetime.now()).rstrip('0')

    def render(self):
        """
        Render all the exhibits. To render the book exhibits only (without the
        extended exhibits) run .render_custom('[A-Y]')

        :return:
        """
        self.render_custom('[A-Y]', suffix='book')
        self.render_custom('Z.*', suffix='extended')

    def render_custom(self, *argv, suffix, outdir=None, yaml_header=None):
        """
        Render a custom list of exhibits using standard glob file expansion.

        The book exhibits are A-Y with Tn and Tg net and gross.

        All extended exhibits are Z...

        :param argv:
        :return:
        """
        base_dir0 = Path.home() / 'aggregate/cases'
        base_dir = base_dir0 / self.case_object.case_id
        if base_dir.exists() is False:
            raise ValueError('{self.case_id} directory not found')

        # this is the actual content
        blobs = []
        ids = []
        for pattern in argv:
            for p in sorted(base_dir.glob('*.md')):
                if re.match(pattern, p.stem):
                    ids.append(p.stem)
                    blobs.append(p.read_text(encoding='utf-8'))

        desc = [f'{self.case_object.case_description}\n']
        spec = self.case_object.to_dict()
        desc.append('### Distributions\n\n```aggregate')
        desc.append(f'# Line A (usually thinner tailed)')
        desc.append(f'{spec["a_distribution"]}\n')
        desc.append(f'# Line B Gross (usually thicker tailed)')
        desc.append(f'{spec["b_distribution_gross"]}\n')
        desc.append(f'# Line B Net')
        desc.append(f'{spec["b_distribution_net"]}\n```\n')
        desc.append(f'### Other Parameters\n')
        # desc.append('<ul>')
        desc.append(f'* `reg_p = {spec["reg_p"]}`')
        desc.append(f'* `roe = {spec["roe"]}`')
        desc.append(f'* `d2tc = {spec["d2tc"]}`')
        desc.append(f'* `s_values = {spec["s_values"]}`')
        desc.append(f'* `gs_values = {spec["gs_values"]}`')
        desc.append(f'* `f_discrete = {spec["f_discrete"]}`')
        desc.append(f'* `log2 = {spec["log2"]}`')
        desc.append(f'* `bs = {spec["bs"]}`')
        desc.append(f'* `padding = {spec["padding"]}`\n')
        if suffix == 'book':
            desc.append('## Description of Tables and Figures {#sec-desc}\n\n')
        desc = '\n'.join(desc)

        # hand create the markdown
        sio = StringIO()
        if yaml_header is not None:
            sio.write(yaml_header)
        sio.write(f'# {self.case_object.case_name} {suffix.title()} Case Study Results\n\n')
        if suffix == 'book':
            sio.write('## Exhibits by Chapter\n')
            sio.write('* Chapter 2: Basic loss statistics (A-C)\n')
            sio.write('* Chapter 4: VaR, TVaR and EPD statistics (D, E)\n')
            sio.write('* Chapter 7: Portfolio pricing, used for calibration (F, G)\n')
            sio.write('* Chapter 9: Classical portfolio and stand-alone pricing (H-L)\n')
            sio.write('* Chapter 11: Modern portfolio and stand-alone pricing (M-Q)\n')
            sio.write('* Chapter 13: Classical allocations (R, S)\n')
            sio.write('* Chapter 15: Modern allocations (T-Y)\n\n')
            sio.write('See @sec-desc for more details.\n\n')
        elif suffix == 'extended':
            sio.write('Supplemental tables and graphs.\n')
        for blob in blobs:
            sio.write('\n')
            sio.write(blob)
            sio.write('\n\n')
        sio.write(f'## {self.case_object.case_name} Case Description\n\n')
        sio.write(desc)
        if suffix == 'book':
            self.add_description(sio)

        # rebase images, entered in snippets as /{case_id}/...
        # other rebasing issues removed because we use the custom installed template
        h = sio.getvalue()
        if outdir is None:
            p = Path.home() / f'aggregate/cases'
            h = h.replace(f'/{self.case_object.case_id}', f'{self.case_object.case_id}')
        else:
            p = Path(outdir)
            h = h.replace(f'/{self.case_object.case_id}', self.case_object.case_id)
            # also copy the image files to img
            img = p / self.case_object.case_id
            img.mkdir(exist_ok=True, parents=True)
            for i in sorted(base_dir.glob('*.png')):
                if re.match(pattern, i.stem):
                    l = img / i.name
                    if l.exists():
                        l.unlink()
                    l.hardlink_to(i)
        p = p / f'{self.case_object.case_id}_{suffix}.md'
        p.write_text(h, encoding='utf-8')
        logger.info(f'Rendered {len(blobs)} exhibits and plots.')

    @staticmethod
    def add_description(sio):
        """
        Add description of exhibits to sio

        """

        txt = """
    
| Ref. |  Kind  | Chapter | Number(s)             | Description                                                                      |
|:----:|:------:|:-------:|:----------------------|:---------------------------------------------------------------------------------|
|  A   | Table  |    2    | 2.3, 2.5, 2.6, 2.7    | Estimated mean, CV, skewness  and kurtosis by line and in  total, gross and net. |
|  B   | Figure |    2    | 2.2, 2.4, 2.6         | Gross and net densities on a linear and log scale.                               |
|  C   | Figure |    2    | 2.3, 2.5, 2.7         | Bivariate densities: gross and net with gross sample.                            |
|  D   | Figure |    4    | 4.9, 4.10, 4.11, 4.12 | TVaR, and VaR for unlimited and limited variables, gross and net.                |
|  E   | Table  |    4    | 4.6, 4.7, 4.8         | Estimated VaR, TVaR, and EPD by line and in total, gross, and net.               |
|  F   | Table  |    7    | 7.2                   | Pricing summary.                                                                 |
|  G   | Table  |    7    | 7.3                   | Details of reinsurance.                                                          |
|  H   | Table  |    9    | 9.2, 9.5, 9.8         | Classical pricing by method.                                                     |
|  I   | Table  |    9    | 9.3, 9.6, 9.9         | Sum of parts (SoP) stand-alone vs. diversified classical pricing by method.      |
|  J   | Table  |    9    | 9.4, 9.7, 9.10        | Implied loss ratios from classical pricing by method.                            |
|  K   | Table  |    9    | 9.11                  | Comparison of stand-alone and sum of parts premium.                              |
|  L   | Table  |    9    | 9.12, 9.13, 9.14      | Constant CoC pricing by unit for Case Study.                                     |
|  M   | Figure |   11    | 11.2, 11.3, 11.4,11.5 | Distortion envelope for Case Study, gross.                                       |
|  N   | Table  |   11    | 11.5                  | Parameters for the six SRMs and associated distortions.                          |
|  O   | Figure |   11    | 11.6, 11.7, 11.8      | Variation in insurance statistics for six distortions  as $s$ varies.            |
|  P   | Figure |   11    | 11.9, 11.10, 11.11    | Variation in insurance statistics as the asset limit is varied.                  |
|  Q   | Table  |   11    | 11.7, 11.8, 11.9      | Pricing by unit and distortion for Case Study.                                   |
|  R   | Table  |   13    | 13.1 missing          | Comparison of gross expected losses by Case, catastrophe-prone lines.            |
|  S   | Table  |   13    | 13.2, 13.3, 13.4      | Constant 0.10 ROE pricing for Case Study, classical PCP methods.                 |
|  T   | Figure |   15    | 15.2 - 15.7 (G/N)     | Twelve plot.                                                                     |
|  U   | Figure |   15    | 15.8, 15.9, 15.10     | Capital density by layer.                                                        |
|  V   | Table  |   15    | 15.35, 15.36, 15.37   | Constant 0.10 ROE pricing for Cat/Non-Cat Case Study, distortion, SRM methods.   |
|  W   | Figure |   15    | 15.11                 | Loss and loss spectrums.                                                         |
|  X   | Figure |   15    | 15.12, 15.13, 15.14   | Percentile layer of capital  allocations by asset level.                         |
|  Y   | Table  |   15    | 15.38, 15.39, 15.40   | Percentile layer of capital  allocations compared to distortion allocations.     |

"""
        sio.write(txt)
