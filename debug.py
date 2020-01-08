# generic file set up to debug aggregate

# common header for smve37
import sys

import aggregate as agg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import itertools
from importlib import reload
import logging
import inspect
import re

pd.set_option('display.max_rows', 500)

# set up logs
aggdevlog = logging.getLogger('aggdev.log')
log = logging.getLogger('aggregate')
log.setLevel(logging.WARNING)

plt.rcParams.update({'font.size': 7})

uw = agg.Underwriter(create_all=False, update=False)

def run_test():

    port = uw('''port Test1
        agg A  8 claims sev lognorm 10 cv 1 poisson
        agg B 80 claims sev lognorm 1 cv 0.1 poisson
    ''')

    port.update(bs=1 / 32, log2=13, remove_fuzz=True, add_exa=True, padding=2)

    port.audits()
    plt.show()

    ans = port.gradient()

if __name__ == '__main__':
    run_test()
