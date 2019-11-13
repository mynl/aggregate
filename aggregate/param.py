"""
Functions to determine and setup parameters
"""

import numpy as np
import pandas as pd
from IPython.core.display import display
## TODO FIX yaml issues
## from ruamel import yaml


def hack_make_lines_from_csv(fn='../../../data/DIRECT_IEE.csv', do_save=False):
    """
    placeholder
    make industry lines from IEE extract
    provenance: IND_IEE*....py in python

    :param do_save:
    :param fn:
    :return:
    """
    D = pd.read_csv(fn)
    line_to_pers_comm = {'Aggregate write-ins for Other Lines of Business': 'Commercial',
                         'Aircraft (all perils)': 'Commercial',
                         'Allied Lines': 'Commercial',
                         'Boiler and Machinery': 'Commercial',
                         'Burglary and theft': 'Commercial',
                         'Commercial Auto Liability': 'Commercial',
                         'Commercial Auto Physical Damage': 'Commercial',
                         'Commercial Multiple Peril (Liability Portion)': 'Commercial',
                         'Commercial Multiple Peril (Non-Liability Portion)': 'Commercial',
                         'Credit': 'Commercial',
                         'Credit A & H': 'Commercial',
                         'Earthquake': 'Commercial',
                         "Excess workers' compensation": 'Commercial',
                         'Farmowners Multiple Peril': 'Personal',
                         'Federal Flood': 'Commercial',
                         'Fidelity': 'Commercial',
                         'Financial Guaranty': 'Commercial',
                         'Fire': 'Commercial',
                         'Group A&H (See Interrogatory 1)': 'Commercial',
                         'Homeowners Multiple Peril': 'Personal',
                         'Inland Marine': 'Commercial',
                         'International': 'Commercial',
                         'Medical Professional Liability': 'Commercial',
                         'Mortgage Guaranty': 'Commercial',
                         'Multiple Peril Crop': 'Commercial',
                         'Ocean Marine': 'Commercial',
                         'Other A&H (See Interrogatory 1)': 'Commercial',
                         'Other Liability - Claims-made': 'Commercial',
                         'Other Liability - Occurrence': 'Commercial',
                         'Private Crop': 'Commercial',
                         'Private Passenger Auto Liability': 'Personal',
                         'Private Passenger Auto Physical Damage': 'Personal',
                         'Products Liability': 'Commercial',
                         'Reinsurance-Nonproportional Assumed': 'Commercial',
                         'Surety': 'Commercial',
                         'TOTAL NET': 'DROP',
                         'TOTALS DIRECT and ASSUMED': 'DROP',
                         'TOTALS DIRECT and PROPORTIONAL ASSUMED (Lines 1 through 34)': 'DROP',
                         'Warranty': 'Commercial',
                         "Workers' Compensation": 'Commercial'}

    line_to_prop_auto_liab = {'Aggregate write-ins for Other Lines of Business': 'Liability',
                              'Aircraft (all perils)': 'Liability',
                              'Allied Lines': 'Property',
                              'Boiler and Machinery': 'Property',
                              'Burglary and theft': 'Property',
                              'Commercial Auto Liability': 'Auto',
                              'Commercial Auto Physical Damage': 'Auto',
                              'Commercial Multiple Peril (Liability Portion)': 'Liability',
                              'Commercial Multiple Peril (Non-Liability Portion)': 'Property',
                              'Credit': 'Liability',
                              'Credit A & H': 'Liability',
                              'Earthquake': 'Property',
                              "Excess workers' compensation": 'Liability',
                              'Farmowners Multiple Peril': 'Property',
                              'Federal Flood': 'Property',
                              'Fidelity': 'Liability',
                              'Financial Guaranty': 'Liability',
                              'Fire': 'Property',
                              'Group A&H (See Interrogatory 1)': 'Liability',
                              'Homeowners Multiple Peril': 'Property',
                              'Inland Marine': 'Property',
                              'International': 'Liability',
                              'Medical Professional Liability': 'Liability',
                              'Mortgage Guaranty': 'Liability',
                              'Multiple Peril Crop': 'Property',
                              'Ocean Marine': 'Liability',
                              'Other A&H (See Interrogatory 1)': 'Liability',
                              'Other Liability - Claims-made': 'Liability',
                              'Other Liability - Occurrence': 'Liability',
                              'Private Crop': 'Property',
                              'Private Passenger Auto Liability': 'Auto',
                              'Private Passenger Auto Physical Damage': 'Auto',
                              'Products Liability': 'Liability',
                              'Reinsurance-Nonproportional Assumed': 'Liability',
                              'Surety': 'Liability',
                              'TOTAL NET': 'DROP',
                              'TOTALS DIRECT and ASSUMED': 'DROP',
                              'TOTALS DIRECT and PROPORTIONAL ASSUMED (Lines 1 through 34)': 'DROP',
                              'Warranty': 'Liability',
                              "Workers' Compensation": 'Liability'}

    line_to_prop_liab = {'Aggregate write-ins for Other Lines of Business': 'Liability',
                         'Aircraft (all perils)': 'Liability',
                         'Allied Lines': 'Property',
                         'Boiler and Machinery': 'Property',
                         'Burglary and theft': 'Property',
                         'Commercial Auto Liability': 'Liability',
                         'Commercial Auto Physical Damage': 'Property',
                         'Commercial Multiple Peril (Liability Portion)': 'Liability',
                         'Commercial Multiple Peril (Non-Liability Portion)': 'Property',
                         'Credit': 'Liability',
                         'Credit A & H': 'Liability',
                         'Earthquake': 'Property',
                         "Excess workers' compensation": 'Liability',
                         'Farmowners Multiple Peril': 'Property',
                         'Federal Flood': 'Property',
                         'Fidelity': 'Liability',
                         'Financial Guaranty': 'Liability',
                         'Fire': 'Property',
                         'Group A&H (See Interrogatory 1)': 'Liability',
                         'Homeowners Multiple Peril': 'Property',
                         'Inland Marine': 'Property',
                         'International': 'Liability',
                         'Medical Professional Liability': 'Liability',
                         'Mortgage Guaranty': 'Liability',
                         'Multiple Peril Crop': 'Property',
                         'Ocean Marine': 'Liability',
                         'Other A&H (See Interrogatory 1)': 'Liability',
                         'Other Liability - Claims-made': 'Liability',
                         'Other Liability - Occurrence': 'Liability',
                         'Private Crop': 'Property',
                         'Private Passenger Auto Liability': 'Liability',
                         'Private Passenger Auto Physical Damage': 'Property',
                         'Products Liability': 'Liability',
                         'Reinsurance-Nonproportional Assumed': 'Liability',
                         'Surety': 'Liability',
                         'TOTAL NET': 'DROP',
                         'TOTALS DIRECT and ASSUMED': 'DROP',
                         'TOTALS DIRECT and PROPORTIONAL ASSUMED (Lines 1 through 34)': 'DROP',
                         'Warranty': 'Liability',
                         "Workers' Compensation": 'Liability'}

    D.loc[:, 'PER_COMM'] = D.FULL_NAME.map(line_to_pers_comm)
    D.loc[:, 'PROP_AUTO_LIAB'] = D.FULL_NAME.map(line_to_prop_auto_liab)
    D.loc[:, 'PROP_LIAB'] = D.FULL_NAME.map(line_to_prop_liab)

    # summarize across lines and including loss adj expense
    line_namer = {'Aggregate write-ins for Other Lines of Business': 'Commercial',
                  'Aircraft (all perils)': 'Commercial',
                  'Allied Lines': 'commprop',
                  'Boiler and Machinery': 'Commercial',
                  'Burglary and theft': 'Commercial',
                  'Commercial Auto Liability': 'cal',
                  'Commercial Auto Physical Damage': 'cal',
                  'Commercial Multiple Peril (Liability Portion)': 'cmp',
                  'Commercial Multiple Peril (Non-Liability Portion)': 'cmp',
                  'Credit': 'Commercial',
                  'Credit A & H': 'Commercial',
                  'Earthquake': 'Commercial',
                  "Excess workers' compensation": 'Commercial',
                  'Farmowners Multiple Peril': 'Personal',
                  'Federal Flood': 'Commercial',
                  'Fidelity': 'Commercial',
                  'Financial Guaranty': 'Commercial',
                  'Fire': 'commprop',
                  'Group A&H (See Interrogatory 1)': 'Commercial',
                  'Homeowners Multiple Peril': 'ho',
                  'Inland Marine': 'im',
                  'International': 'Commercial',
                  'Medical Professional Liability': 'med mal',
                  'Mortgage Guaranty': 'Commercial',
                  'Multiple Peril Crop': 'Commercial',
                  'Ocean Marine': 'Commercial',
                  'Other A&H (See Interrogatory 1)': 'Commercial',
                  'Other Liability - Claims-made': 'Commercial',
                  'Other Liability - Occurrence': 'Commercial',
                  'Private Crop': 'Commercial',
                  'Private Passenger Auto Liability': 'ppa',
                  'Private Passenger Auto Physical Damage': 'ppa',
                  'Products Liability': 'Commercial',
                  'Reinsurance-Nonproportional Assumed': 're assumed',
                  'Surety': 'Commercial',
                  'TOTAL NET': 'DROP',
                  'TOTALS DIRECT and ASSUMED': 'total',
                  'TOTALS DIRECT and PROPORTIONAL ASSUMED (Lines 1 through 34)': 'DROP',
                  'Warranty': 'Commercial',
                  "Workers' Compensation": 'wc'}

    def clean(D):
        D['DIL'] = (D.IL + D.DCC + D.AOE)
        D['E'] = pd.to_numeric(D.CUSTOMER, errors='coerce') + + pd.to_numeric(D.PAPER, errors='coerce')
        df = D[['FULL_NAME', 'PROP_AUTO_LIAB', 'YEAR', 'DEP', 'DIL', 'E']].copy()
        df = df.dropna()
        test_lines = ['Fire', 'Allied Lines',
                      'Homeowners Multiple Peril',
                      'Medical Professional Liability',
                      'Commercial Multiple Peril (Non-Liability Portion)',
                      'Commercial Multiple Peril (Liability Portion)',
                      'Inland Marine',
                      "Workers' Compensation",
                      'Private Passenger Auto Liability',
                      'Commercial Auto Liability',
                      'Private Passenger Auto Physical Damage',
                      'TOTALS DIRECT and ASSUMED']
        test_lines = [f'"{i}"' for i in test_lines]
        query = f' FULL_NAME in [{", ".join(test_lines)}]'
        test_df = df.query(query)
        test_df['NAME'] = test_df.FULL_NAME.map(line_namer)
        test_df = test_df.drop('FULL_NAME', axis=1)
        test_df = test_df.groupby(['NAME', 'YEAR']).sum()

        test_df['LR'] = test_df.DIL.divide(test_df.DEP)
        test_df['ER'] = test_df.E.divide(test_df.DEP)
        test_df['CR'] = test_df.LR + test_df.ER
        test_df['PLR'] = test_df.LR.divide(1 - test_df.ER)
        test_df = test_df.reset_index()
        return test_df

    def cov(x):
        return x.std() / x.mean()

    def skew(x):
        return x.skew()

    def premium_net_of_expenses(x):
        """
        premium net of expenses

        """
        return x.iloc[-1]

    sf = [np.mean, np.std, cov, skew]
    tl = clean(D)
    piv = tl.groupby('NAME').agg({'LR': sf, 'PLR': sf, 'ER': sf, 'CR': sf, 'DEP': premium_net_of_expenses})
    display(piv)
    piv.plot(kind='bar', subplots=True, figsize=(9, 8), layout=(-1, 4), sharex=True)
    ans = piv['DEP']
    ans['PLRCV'] = piv[('PLR', 'cov')]
    ans['ER'] = piv[('ER', 'mean')]
    ans['NEP'] = ans.EP * (1 - ans.ER)
    ans['esev'] = [40000, 100000, 250000, 15000, 50000, 750000, 15000, 15000, 10000]
    ans['cvsev'] = 0.5
    ans['limit'] = 3000000
    # ans['limit'] = [3000000, 2000000, 5000000, 1000000, 2000000, 500000, 10000000]

    d = dict()
    for name, l in ans.iterrows():
        s = dict(name='lognorm', mean=float(l['esev']), cv=float(l['cvsev']))
        f = dict(n=float(l.NEP * 1000 / l.esev), contagion=float(l.PLRCV ** 2))
        e = dict(name=name, severity=s, frequency=f)
        d[f'ind {name}'] = e
    print(yaml.dump(d, default_flow_style=False, indent=4))
    if do_save:
        with open('./agg/aggregate.yaml', 'a') as f:
            yaml.dump(d, stream=f, default_flow_style=False, indent=4)
    return tl
