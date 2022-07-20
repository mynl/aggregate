# taken from book python CH12_Reservess2.ipynb and common_scripts2022
# here is just the code you need to make the reserve stories work stand alone
# with the new portfolios
# TODO: convert to new, just HTML style output and generic reporting
# make the md stories separate text aside from the exhibits

from jinja2 import Environment, FileSystemLoader, select_autoescape
from aggregate import Distortion
import numpy as np
import pandas as pd
from pathlib import Path

# template_folder = 'c:\\s\\telos\\python\\aggregate_project\\examples\\reserves\\templates'
# env = Environment(loader=FileSystemLoader(template_folder),
#                                autoescape=select_autoescape(['html', 'xml']))

template_folder = Path(__file__) / '../../../aggregate/templates'
print(template_folder.resolve(), template_folder.exists())

env = Environment(loader=FileSystemLoader(template_folder),
                               autoescape=select_autoescape(['html', 'xml']))


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


def year_end_option_analysis(self, paid_line, a=0, p=0, resolution_names=None):
    """
    reserve story: what happens at the end of the year?

    includes detailed_loss_story and ro_gc_compare
    includes margin_earned

    """

    a, p = self.set_a_p(a, p)

    # how margin is earned
    n = len(self.line_names)
    implied_periods = [f'{i + 1}' for i in range(n)]
    if resolution_names is None:
        resolution_names = [f'In prd {i + 1}' for i in range(n)]
    me = self._raw_premium_capital.loc['T.M'].iloc[:n].to_numpy()
    self.EX_margin_earned = pd.DataFrame({'Period ending': implied_periods,
                                          'Beginning risk margin': np.cumsum(me[::-1])[::-1],
                                          'Margin earned': me},
                                         index=resolution_names,
                                         )
    self.EX_margin_earned.index.name = 'Resolution during period'
    self.EX_margin_earned.loc['Total', :] = ['All', np.sum(np.cumsum(me[::-1])), np.sum(me)]
    self.me = me  # for risk tenor

    ##
    # convenience
    self.bit = self.augmented_df.filter(regex=f'loss|p_|F|S|lev|exag?(_ημ)?_({self.line_name_pipe})$').copy()
    self.last_a = -1

    # add all the totals for reserves (was make_stand_alone)
    # compute stand alone LEV and LEVg for paid_line. Note LEV is already
    # computed but this is a good check
    # append to self.bit that needs to exist already as an extract from adf
    # add relevant computed parts to bit
    if paid_line[0:2] == 'ημ':
        ser = self.augmented_df[paid_line]
        # stand alone run off assets
        a_ro = self.density_df.loss.iloc[np.argmax(ser.cumsum() > p)]
    else:
        ser = self.augmented_df[f'p_{paid_line}']
        # stand alone run off assets
        li = self.line_names.index(paid_line)
        self.ag = self.agg_list[li]
        a_ro = self.ag.q(p)

    # stand alone S and LEV, LEVg for the reserve paid_line, shift up: S is > x
    S = ser.shift(-1, fill_value=0)[::-1].cumsum()
    LEV = S[::-1].shift(1, fill_value=0).cumsum() * self.bs
    gS = np.array(self.distortion.g(S[::-1]))  # some dist return np others don't this converts to numpy...
    LEVg = pd.Series(np.hstack((0, gS[:-1])).cumsum() * \
                     self.bs, index=ser.index)

    self.bit[f'S_{paid_line}'] = S
    self.bit[f'levg_{paid_line}'] = LEVg

    # run off assets
    mlv_ro = LEVg[a_ro]

    # ro = stand alone
    self.bit[f'ro_eva_{paid_line}'] = LEVg - mlv_ro
    self.bit[f'ro_da_{paid_line}'] = a_ro - self.bit.loss
    self.bit[f'ro_dmv_{paid_line}'] = self.bit[f'ro_eva_{paid_line}'] + self.bit[f'ro_da_{paid_line}']

    # premium in going concern
    p_0 = self.bit.loc[a, f'exag_{paid_line}']
    p_total = self.bit.loc[a, f'exag_total']
    p1 = p_total - p_0
    # gc = going concern view
    self.bit[f'gc_da_{paid_line}'] = a - p1 - self.bit.loss
    self.bit[f'gc_dmv_{paid_line}'] = (a - p_total) - (self.bit.loss - LEVg)
    self.bit[f'gc_eva_{paid_line}'] = self.bit[f'gc_dmv_{paid_line}'] - self.bit[f'gc_da_{paid_line}']
    self.bit.index.name = 'Ending a'

    # based on gc_ro_compare from runoff
    #
    # part of bit to display (detailed_story_results=display part of bit) [two step does not use ημ ]
    ix = np.searchsorted(self.bit[f'gc_eva_{paid_line}'], 0)
    break_even = self.bit.iloc[ix, 0]
    # quasi steady state view of ye assets
    est_paid = self.density_df.loc[a, f'exa_{paid_line}']
    # make ye assets that is in the index
    est_ye_assets = self.q(self.cdf(a - est_paid))
    # range of reasonable estimates, all index values
    idx = [8532.0,
9532.0,
10743.5,
10848.0,
13164.0,
15480.0,
18000.0,break_even, est_ye_assets, a]
    # now have included a_x in the index extracted:
    rogc = self.bit.loc[idx,
                        ['S', f'S_{paid_line}',
                         f'lev_{paid_line}', f'levg_{paid_line}',
                         f'exa_{paid_line}', f'exag_{paid_line}',
                         'exa_total', 'exag_total',
                         f'ro_da_{paid_line}', f'ro_dmv_{paid_line}', f'ro_eva_{paid_line}',
                         f'gc_da_{paid_line}', f'gc_dmv_{paid_line}', f'gc_eva_{paid_line}']]. \
        fillna('').T
    # compute market value and put in the right spot
    rogc.loc['mv'] = rogc.columns - rogc.loc[f'levg_{paid_line}']
    mv_gc = a - rogc.loc['exag_total'].iloc[-1]
    rogc.loc['mv', a] = mv_gc
    rogc.loc['Assets'] = rogc.columns
    rogc.columns = list(rogc.columns)[:-3] + ['Break Even', 'Est. EoY', 'BoY']
    rogc.index.name = "Item and View"
    # rogc = rogc.iloc[[15, 0,1,2,3,4,5,6,7,14,8,10,11,13]]
    self._rogc = rogc
    self.EX_year_end_option_analysis = rogc.rename(index=self.renamer)

    # -99 a_x0?
    return -99, est_paid, est_ye_assets, mv_gc, break_even


def reserve_story_md(self, paid_line, a=0, p=0, ROE=0, template='html'):
    """
    pull out relevant values and generate markdown file for the reserve_story template

    note p and ROE are fixed once you calibrate the distortions
    so they are not input options here

    makes a nice load of exhibits....

    There were two additional templates reserve_two_step and reserve_runoff
    The first did a two step: resolve G, resolve loss model
    Run off did a four step G, n, reserves, loss model.
    They included a subset of these exhibits.

    Original version used markdown template files. (Originals are in the templates folder)

    :param template:
    :param line:

    """
    global env

    if template == 'html':
        template = 'reserve_story.html'
    else:
        template = 'reserve_story.md'
    # last applied distortion
    dist_name = self.distortion.name
    a, p = self.set_a_p(a, p)

    # update exhibits...make_all makes premium_capital and acc_eco_bs
    self.make_all(p=p)
    # compute all the custom values
    a_x0, est_paid, est_ye_assets, mv_gc, \
    break_even =  year_end_option_analysis(self, a=a, p=p, paid_line=paid_line)
    print(a_x0, est_paid, f'{est_paid:,g}', est_ye_assets, mv_gc, break_even)

    # make the jinja template
    t = env.get_template(template)
    # fill in values and render

    def f(x):
        try:
            if abs(x) < 3:
                return f'{x:.6g}'
            else:
                return f'{x:,.2f}'
        except:
            return x

    def qs(df):

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
            'props': f'text-align: right;'
        }
        all_styles = [cell_hover, index_names, headers, center_heading,  left_index, td]
        return df.style.set_table_styles(all_styles).format(f).to_html()

    # nice extract from dist_ans
    distortion_information = self.dist_ans.reset_index(drop=False). \
        sort_values('method')[['method', 'param']]. \
        rename(columns=dict(method='Distortion', param='Shape Parameter')). \
        set_index('Distortion').rename(index=Distortion._distortion_names_)

    self.md = t.render(
        scenario_name='story',
        program=self.nice_program(),
        p=f'{p:.4g}',
        ROE=f'{ROE * 100:.1f}',
        dist=dist_name,
        a_x0=a_x0, # stand alone capital
        a_x=a,
        a_x1=self.agg_list[self.line_names.index(paid_line)].q(p),
        ex0=self.audit_df.loc['total', 'Mean'],
        est_paid=f'{est_paid:,g}',
        est_ye_assets=f'{est_ye_assets:,g}',
        mv_gc=f'{mv_gc:.1f}',
        break_even=break_even,
        loss_threshold=a_x0 - break_even,
        mvp_ro=f(self._rogc.loc['levg_Xm1', 'BoY']), # self.a_x0]),
        mvp_a=f(self._rogc.loc['levg_Xm1', 'BoY']),
        mvp_gc=f(self._rogc.loc['exag_Xm1', 'BoY']), # self.a_x]),
        basic_loss_statistics=qs(self._report_df),
        distortion_information=qs(distortion_information),
        premium_capital=qs(self.EX_premium_capital),
        margin_earned=qs(self.EX_margin_earned),
        accounting_economic_balance_sheets=qs(self.EX_accounting_economic_balance_sheet),
        year_end_option_analysis=qs(self.EX_year_end_option_analysis)
    )

