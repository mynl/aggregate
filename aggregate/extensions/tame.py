# make the tame case study 

from aggregate.extensions import case_studies as cs
from aggregate import build

if __name__ == '__main__':

    # calibrate reinsurance
    recalc = build('agg B 1 claim sev gamma 50 cv 0.15 fixed',
                   log2=16, bs=1/64)
    a, d = recalc.q(0.8), recalc.q(0.99)
    y = d - a

    # create exhibits
    tame = cs.CaseStudy()
    tame.factory(case_id='tame',
                 case_name='Tame',
                 case_description='Tame Case in the new syntax.',
                 a_distribution='agg A 1 claim sev gamma  50 cv 0.10 fixed',
                 b_distribution_gross='agg B 1 claim sev gamma  50 cv 0.15 fixed',
                 b_distribution_net=f'agg B 1 claim sev gamma  50 cv 0.15 fixed aggregate net of {y} xs {a}',
                 reg_p=0.9999,
                 roe=0.10,
                 d2tc=0.3,
                 f_discrete=False,
                 s_values=[.005, 0.01, 0.03],
                 gs_values=[0.029126,   0.047619,   0.074074],
                 bs=1/64,
                 log2=16,
                 padding=1)
    tame.full_monty()
    tame.to_json()
    tame.browse_exhibits()
