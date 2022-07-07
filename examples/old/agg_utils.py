import pandas as pd
import aggregate as agg
import numpy as np


def extract_info_fixed(port, p, r, dd=None):
    """
    Extract info from portfolio object port for TVaR threshold p
    Price using Wang and PH transforms to overall ROE of r (upto the TVaR threshold)

    FIXED calibration

    """
    df = port.density_df
    bs = df.index[1]
    var = port.q(p)
    tvar = df.loc[var, 'exgta_total']
    tvar0 = df.index.get_loc(float(tvar), method='nearest')
    tvar0 = df.index[tvar0]
    bit1 = df.filter(regex='S|exa_').loc[tvar-bs/2:tvar+bs/2, :]
    tvar = bit1.index[0]
    bit2 = df.filter(regex='S|exgta_').loc[var, :]
    ans = pd.concat((bit1, pd.DataFrame(bit2).T), axis=0, sort=True).sort_index().fillna('')
    ans.index.name = 'a'
    ans2 = pd.DataFrame({**dict(var=var, tvar=tvar),
                         **{'S(var)': ans.iloc[0,0], 'S(tvar)': ans.iloc[1,0]},
                         **{f'E({i}|X>var)': ans.loc[var, f'exgta_{i}'] for i in port.line_names_ex},
                         **{f'EL({i})': ans.loc[tvar, f'exa_{i}'] for i in port.line_names_ex},
                         }, index=pd.MultiIndex.from_arrays([[port.name.replace("~", ' ')], ['Ind Std']]))
    # Ind Std = industry standard approach = "basic" here
    for l in port.line_names_ex:
        el = ans2.loc[:, f'EL({l})']
        t = ans2.loc[:, f'E({l}|X>var)']
        P_basic = (r*t + el) / (1+r)
        ans2[f'mA_Prem({l})'] = P_basic
        ans2[f'mA_LR({l})'] = el / P_basic
    ans2 = ans2.T.copy()
    # now add PH and wang pricing...
    ans3 = ans2.copy()
    for dist in ['ph', 'wang']:
        ans3.columns = pd.MultiIndex.from_arrays([[port.name.replace("~", ' ')], [dist]])
        if dd is None:
            # recalibrate
            cd = port.calibrate_distortions(ROEs=[r], As=[tvar])
            dd = agg.Distortion.distortions_from_params(cd, cd.loc[(slice(None), slice(None), dist), :].
                                                        index[0][0:2], plot=False)
        rho = dd[dist].shape
        ans_table, ans_stacked = port.apply_distortion(dd[dist], num_plots=0)
        ans3.loc['distn param', :] = rho
        # strip out new prem from ans_table
        for l in port.line_names_ex:
            el = ans3.loc[f'EL({l})', :]
            P_basic = ans_table.loc[tvar, f'exag_{l}']
            ans3.loc[f'mB_Prem({l})', :] = P_basic
            ans3.loc[f'mB_LR({l})', :] = el / P_basic
            # for method A the premium is different
            # this reproduces what comes from apply distortion...which it does
            # exleaUC = port.cumintegral(ans_table[f'exeqa_{l}'] * ans_table.gp_total, 1)  # unconditional
            # exixgtaUC = np.cumsum(
            #     ans_table.loc[::-1, f'exeqa_{l}'] / ans_table.loc[::-1, 'loss'] *
            #     ans_table.loc[::-1, 'gp_total'])
            # ans_table[f'mA_exag_{l}'] = exleaUC + exixgtaUC * df.loss
            # this actually implements method A...difference is in the gtaUC part...
            exleaUC = port.cumintegral(ans_table[f'exeqa_{l}'] * ans_table.gp_total, 1)  # unconditional
            # exixgtaUC = df.loc[:, f'exgta_{l}'] * ans_table.loc[:, 'gS']
            ans_table['temp'] = exleaUC # to enable indexed lookup
            i = tvar0  # np.round(tvar0-bs, decimals=2)
            ans3.loc[f'mA_Prem({l})', :] = ans_table.loc[i, 'temp'] + df.loc[var, f'exgta_{l}'] * ans_table.loc[i, 'gS']
            ans3.loc[f'mA_LR({l})', :] = el / ans3.loc[f'mA_Prem({l})', :]
        wA=float(ans3.loc['mA_Prem(A)', :])
        wB=float(ans3.loc['mA_Prem(B)', :])
        wt=float(ans3.loc['mA_Prem(total)', :])
        print(f'dist {dist:<4s}, prems {wA:.2f}, {wB:.2f}, {wt:.2f}, err = {wt-wA-wB:.2f}')
        ans2 = pd.concat((ans2, ans3), axis=1, sort=True)
    ans2.loc['r', :] = r
    ans2 = ans2.sort_index(ascending=True)
    return ans2, dd


def story():
    """
    Explain the result for the different allocations of premium...
    :return:
    """
    s = """
    
    
    """
    return s


# junk
# # proof of extract_info_fixed...
# r = 0.125
# v = 1/(1+r)
# d = r * v
#
# var = xx0.q(0.99)
# tvar = xx0.TVaR(0.99)
# tvar0 = xx0.density_df.index.get_loc(float(tvar), method='nearest')
# tvar0 = xx0.density_df.index[tvar0]
#
# bs = xx0.density_df.loss.iloc[1]
# # allocation of TVaR comes from exgta(var)
# cap_alloc = xx0.density_df.filter(regex='exgta').loc[var, :]
# el_eq_pri = xx0.density_df.filter(regex='exa').loc[tvar0, :]
# #method_prem_line
# A_prem_A = el_eq_pri.exa_A * v + cap_alloc.exgta_A * d
# A_prem_B = el_eq_pri.exa_B * v + cap_alloc.exgta_B * d
# A_prem_t = el_eq_pri.exa_total * v + cap_alloc.exgta_total * d
# print(A_prem_A, A_prem_B, A_prem_t, A_prem_A+ A_prem_B - A_prem_t)
# print(var, tvar, tvar0)
# print(cap_alloc)
# print(el_eq_pri)
# print(f'tvar allocation error {cap_alloc.exgta_A + cap_alloc.exgta_B - cap_alloc.exgta_total}')
