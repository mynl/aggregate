# scratch file

# from sly import Laxer, Parser



import sys
import aggregate as agg
from aggregate import build
from aggregate.parser import run_one
from aggregate.utils import make_ceder_netter
from numpy import exp
from scipy.interpolate import interp1d

uw = agg.Underwriter(create_all=True)
logger_level(20)


if 0:
    gross = uw(
        'agg HU 2 claims 1200000000 x 0 sev 0.6590540043511113 @ lognorm 2.5 poisson')
    neto = uw('agg HU 2 claims 1200000000 x 0 sev 0.6590540043511113 @ lognorm 2.5 net of 372.4 xs 40.25 occurrence poisson')
    neta = uw('agg HU 2 claims 1200000000 x 0 sev 0.6590540043511113 @ lognorm 2.5 poisson net of 372.4 xs 40.25 aggregate')

    gross.easy_update(bs=0.25, log2=19, padding=1)
    neto. easy_update(bs=0.25, log2=19, padding=1)
    neta. easy_update(bs=0.25, log2=19, padding=1)

    frame = {'gross': gross, 'neto': neto, 'neta': neta}

    ans = []
    for k, ob in frame.items():
        a = ob.q(0.999)
        ans.append(pd.Series([a, ob.density_df.at[a, 'exa']],
                             index=['assets', 'exa']))

    display(pd.concat(ans, axis=1, keys=frame.keys()))

    # these are numbers from the new_case_studies md files
    freq1, sigma1 = 70, 1.9
    freq2, sigma2 = 2, 2.5
    mu1 = np.log(70 / freq1) - sigma1**2 / 2
    mu2 = np.log(30 / freq2) - sigma2**2 / 2
    print(freq1, exp(mu1), sigma1, freq2, exp(
        mu2), sigma2, mu1, sigma1, mu2, sigma2)


    bookg = uw(f'''
    port Gross_hs
        agg SCS 70 claims 1200000000 x 0 sev 0.1644744565771549 @ lognorm 1.9 poisson
        agg HU 2 claims 1200000000 x 0 sev 0.6590540043511113 @ lognorm 2.5  poisson
    ''', create_all=False)

    bookn = uw(f'''
    port Net_hs
        agg SCS 70 claims 1200000000 x 0 sev 0.1644744565771549 @ lognorm 1.9 poisson
        agg HU 2 claims 1200000000 x 0 sev 0.6590540043511113 @ lognorm 2.5 poisson net of 372.4 xs 40.25 aggregate
    ''', create_all=False)

    bookg.update(bs=0.25, log2=19, padding=1, remove_fuzz=True)

    bookn.update(bs=0.25, log2=19, padding=1, remove_fuzz=True)

    for ob in [bookg, bookn, gross, neto, neta]:
        display(ob)
