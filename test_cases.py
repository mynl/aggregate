"""
Test cases for aggregate

E.g. to run from Jupyter enter

!python test_cases.py

"""

import sys
sys.path.append('c:/s/telos/python')

import aggregate as agg
from aggregate import Aggregate, Severity
import warnings
import numpy as np
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    ex = agg.Example()
    port = ex['Three Line Example']
    assert port.audit_df.MeanErr.abs().sum() < 1e-5
    assert port.audit_df.CVErr.abs().sum() < 5e-5
    a, p, test, params, dd, table, stacked = port.uat()
    assert a['lr err'].abs().sum() < 1e-8
    assert np.all(test.filter(regex='err[_s]', axis=1).abs().sum() < 1e-8)

    sig = .75
    d1b1 = Aggregate('single', exp_en=10, sev_name='lognorm', sev_a=[sig, sig / 2, sig / 5],
                     sev_scale=100 * np.exp(-sig ** 2 / 2), sev_wt=1, freq_name='poisson', freq_a=0.81)
    xs = np.linspace(0, 1023, 1024)
    df, audit = d1b1.density(xs, verbose=False)
    assert d1b1.stats_total.T.loc['el', :].sum() == 5146.408347357843

    sig = 0.75
    d1b2 = Aggregate('sublines', exp_en=[10, 10, 10, 10], exp_attachment=[0, 0, 100, 50],
                     exp_limit=[np.inf, 100, np.inf, 100], sev_name='lognorm', sev_a=sig,
                     sev_scale=100 * np.exp(-sig ** 2 / 2),
                     sev_wt=1, freq_name='poisson', freq_a=0.81)
    assert d1b2.stats_total.T.loc['el', :].sum() == 6152.9037739696396

    d2b1 = Aggregate('mixed', exp_en=50, exp_attachment=[0, 50, 100, 150, 200], exp_limit=[250, 250, np.inf, 100, 5000],
                     sev_name='lognorm', sev_mean=100, sev_cv=np.linspace(0.05, 2, 5), sev_wt=0.2 * np.ones(5),
                     freq_name='poisson', freq_a=1.2)
    assert d2b1.stats_total.T.loc['el', :].sum() == 11464.315472434911

    fixed = Severity('dhistogram', hxs=[0, 1, 2, 3, 4], hps=[.2, .3, .4, .05, .05])
    assert fixed.moms() == (1.45, 3.1500000000000004, 8.0500000000000007)

    fixed = Severity('chistogram', hxs=[0, 1, 2, 3, 4], hps=[.1, .2, .3, 0, .4])
    assert fixed.moms() == (2.9000000000000004, 10.449999999999999, 41.825000000000003)

    fixed = Severity('fixed', hxs=2)
    assert fixed.moms() == (2, 4, 8)

    fixed = Aggregate('chistogram', exp_en=1, sev_name='dhistogram', sev_xs=[0, 1, 2, 3, 4], sev_ps=.2,
                      freq_name='fixed')
    assert fixed.statistics_df['sev_cv'].sum() == 2.1213203435596428

    # bigger agg example
    sa = Aggregate('my test', exp_en=np.ones(50), exp_premium=0, exp_lr=.75,
                              exp_attachment=20, exp_limit=np.linspace(10, 500, 50), sev_name='lognorm', sev_mean=100, sev_cv=0.8, freq_name='poisson', freq_a=0)

    bs = 1  # 2 and 13
    log2 = 13
    N = 1 << log2
    MAXL = N * bs
    xs = np.linspace(0, MAXL, N, endpoint=False)
    df, audit = sa.density(xs, 1, None, 'exact', sev_calc='discrete', discretization_calc='survival', verbose=True)
    # average of square errors is small:
    np.sum(np.abs(audit.iloc[0:-3,:][ 'rel sev err'])**2)**.5 / (len(audit) - 3)  < 1e-6
    np.sum(np.abs(audit.iloc[0:-3,:][ 'abs sev err'])**2)**.5 / (len(audit) - 3) < 1e-4