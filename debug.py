# generic file set up to debug aggregate

# common header for smve37
# import sys
#
# sys.path.append('c:\\s\\telos\\spectral_risk_measures_monograph\\\Python')
#
# from common_header import *
# import common_scripts as cs
#
#
# pd.set_option('display.max_rows', 500)
#
#
# def run_test():
#     # rt = "tense"
#     rt = "relaxed"
#     # either way need the mass_hints
#     mh = pd.Series([1,0], index=['X:expon', 'Y:uniform'])
#
#     if rt == 'relaxed':
#
#         port = cs.RelaxedPortfolio('''
#
#         port ISA
#             agg X:expon 1 claim sev 1 * expon fixed
#             agg Y:uniform 1 claim sev 3 * uniform fixed
#
#         ''', log2=10, bs=1/64, padding=2)
#
#         a, p = port.set_a_p(0, 0.995)
#         port.calibrate_distortions(Ps=[p], ROEs=[.12], strict=False)
#         print(port.dist_ans)
#         ans2 = port.apply_distortion(port.dists['clin'], mass_hints=mh)
#
#     ans = port.analyze_distortion('clin', A=a, ROE=0.12, plot=True, mass_hints=mh)

import aggregate as agg
from aggregate import build, debug_build
from aggregate.utilities import make_ceder_netter
from examples import case_studies as  cs
from numpy import exp
from examples import case_studies as  cs


def runpf_test():
    case = cs.CaseStudy()
    case.factory_book('cnc')


def run_test():
    prog1 = 'agg parsetest3 5 claims  sev  lognorm 2 cv .3 poisson'
    a = build(prog1)

    prog = 'port PORT0\n\tagg X 1 claim sev lognorm 20 cv .9 poisson'
    port = build(prog)

if __name__ == '__main__':
    run_test()
    # runpf_test()
