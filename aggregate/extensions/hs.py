# make the hu-scs case study

from aggregate.extensions import case_studies as cs
from aggregate import build
import numpy as np

if __name__ == '__main__':

      # parameters from PIR
      freq1, sigma1 = 70, 1.9
      freq2, sigma2 = 2, 2.5
      mu1 = -sigma1**2 / 2
      mu2 = -sigma2**2 / 2
      sev1, sev2 = 1, 15
      print(mu1, sigma1, mu2, sigma2,
            freq1, np.exp(mu1), sigma1,
            freq2, np.exp(mu2), sigma2)

      recalc = build(f'sev Husev {15 * np.exp(mu2)} * lognorm {sigma2}')
      a, d = recalc.isf(0.05), recalc.isf(0.005)
      y = d - a
      print(y, a, d)

      # create exhibits
      hs = cs.CaseStudy()
      hs.factory(case_id='hs',
                 case_name='Hu/SCS Case',
                 case_description='Hu/SCS Case in the new syntax.',
                 a_distribution=f'agg SCS {freq1} claims sev {sev1 * np.exp(mu1)} * lognorm {sigma1} poisson',
                 b_distribution_gross=f'agg Hu {freq2} claims sev {sev2 * np.exp(mu2)} * lognorm {sigma2} poisson',
                 b_distribution_net=f'agg Hu {freq2} claims sev {sev2 * np.exp(mu2)} * lognorm {sigma2} poisson ' \
                                    f'aggregate net of {y} xs {a}',
                 reg_p=0.999,
                 roe=0.10,
                 d2tc=0.3,
                 f_discrete=False,
                 s_values=[.005, 0.01, 0.03],
                 gs_values=[0.029126,   0.047619,   0.074074],
                 bs=1/4,
                 log2=19,
                 padding=1)
      hs.full_monty()
      hs.to_json()
      hs.browse_exhibits()


      hs2 = cs.CaseStudy()
      hs2.factory(case_id='hs_per_occ',
                  case_name='Hu/SCS',
                  case_description='Hu/SCS Case in the new syntax with per occurrence reinsurance .',
                  a_distribution=f'agg SCS {freq1} claims sev {sev1 * np.exp(mu1)}  * lognorm {sigma1} poisson',
                  b_distribution_gross=f'agg Hu {freq2} claims sev {sev2 * np.exp(mu2)} * lognorm {sigma2} poisson',
                  b_distribution_net=f'agg Hu {freq2} claims sev {sev2 * np.exp(mu2)} * lognorm {sigma2} '
                                     f'occurrence net of {y} xs {a} poisson',
                  reg_p=0.999,
                  roe=0.10,
                  d2tc=0.3,
                  f_discrete=False,
                  s_values=[.005, 0.01, 0.03],
                  gs_values=[0.029126,   0.047619,   0.074074],
                  bs=1/4,
                  log2=19,
                  padding=1)
      hs2.full_monty()
      hs2.to_json()
