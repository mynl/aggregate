# make the cat non-cat case study

from aggregate.extensions import case_studies as cs
from aggregate import build


if __name__ == '__main__':
    # calibrate reinsurance
    recalc = build('agg Cat 1 claim sev lognorm 20 cv 1.00 fixed'
                   , log2=16, bs=1/64)
    a, d = recalc.q(0.9), recalc.q(0.995)
    y = d - a

    # create exhibits
    cnc = cs.CaseStudy()
    cnc.factory(case_id='cnc',
                case_name='Cat/Non-Cat',
                case_description='Cat/Non-Cat in the new syntax.',
                a_distribution='agg NonCat 1 claim sev gamma    80 cv 0.15 fixed',
                b_distribution_gross='agg Cat 1 claim sev lognorm  20 cv 1.00 fixed',
                b_distribution_net=f'agg Cat 1 claim sev lognorm  20 cv 1.00 fixed aggregate net of {y} xs {a}',
                reg_p=0.999,
                roe=0.10,
                d2tc=0.3,
                f_discrete=False,
                s_values=[.005, 0.01, 0.03],
                gs_values=[0.029126,   0.047619,   0.074074],
                bs=1/64,
                log2=16,
                padding=1)
    cnc.full_monty()
    cnc.to_json()
    cnc.browse_exhibits()
