# support for Bodoff comparisons

import pandas as pd

def bodoff_exhibit(self, reg_p):
    """
    create the bodoff exhibit for port = self
    at lower reg_p-VaR capital level

    :param self: Portfolio object
    :param reg_p: float, regulatory capital level
    :return: pd.DataFrame of results

    """

    basic = pd.DataFrame(index=pd.Index(['EX', 'sa VaR', 'sa TVaR', 'pct EX', 'coVaR', 'alt coVaR', 'naive coTVaR', 'coTVaR'], name='method'),
                         columns=self.line_names_ex, dtype=float)

    # mean
    basic.loc['EX'] = self.audit_df.Mean.values
    # stand alone
    basic.loc['sa VaR'] = self.var_dict(reg_p, 'lower').values()
    basic.loc['sa TVaR'] = self.var_dict(reg_p, 'tvar').values()
    a = self.q(reg_p, 'lower')

    # pct of loss
    basic.loc['pct EX'] = basic.loc['EX'] / basic.loc['EX', 'total'] * a

    # method 1
    basic.loc['coVaR'] = (self.density_df.loc[a, [f'exeqa_{i}' for i in self.line_names_ex]] / self.density_df.at[a, 'exeqa_total']).values * a

    # method two ...
    basic.loc['alt coVaR'] = (self.density_df.loc[a - self.bs, [f'exi_xgta_{i}' for i in self.line_names] + ['exi_xgta_sum']] * a).values
    # naive tvar view
    basic.loc['naive coTVaR'] = (self.density_df.loc[a - self.bs, [f'exgta_{i}' for i in self.line_names_ex]] / self.density_df.at[a - self.bs, 'exgta_total']).values * a

    # proper co-tvar with calibration
    pt = self.tvar_threshold(reg_p, 'lower')
    # not the generic answer...
    basic.loc['coTVaR'] = self.cotvar(pt)  # list(self.var_dict(pt, 'tvar').values())

    bit = self.density_df[[f'exi_xgta_{i}' for i in self.line_names]].shift(1).cumsum() * self.bs
    bit['total'] = bit.sum(1)

    basic.loc['plc'] = bit.loc[a].values

    return basic

