"""
Test cases for aggregate

E.g. to run from Jupyter enter

!python test_cases.py

"""


import aggregate as agg
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