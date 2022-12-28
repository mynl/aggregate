# make the discrete case study 

from aggregate.extensions import case_studies as cs

if __name__ == '__main__':

    discrete = cs.CaseStudy()
    discrete.factory(case_id='discrete',
                     case_name='Discrete',
                     case_description='PIR Discrete Case Study (no equal points).',
                     a_distribution='agg X1 1 claim dsev [0 8 10] [1/2 1/4 1/4] fixed',
                     b_distribution_gross='agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed',
                     b_distribution_net=f'agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed aggregate net of 70 xs 20',
                     reg_p=1,
                     roe=0.10,
                     d2tc=0.3,
                     f_discrete=True,
                     s_values=[.005, 0.01, 0.03],
                     gs_values=[0.029126,   0.047619,   0.074074],
                     bs=1,
                     log2=8,
                     padding=1)
    discrete.full_monty()
    discrete.to_json()
    discrete.browse_exhibits()

    discrete_eq = cs.CaseStudy()
    discrete_eq.factory(case_id='discrete_equal',
                        case_name='Discrete (equal points)',
                        case_description='PIR Discrete Case Study with equal points.',
                        a_distribution='agg X1 1 claim dsev [0 9 10] [1/2 1/4 1/4] fixed',
                        b_distribution_gross='agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed',
                        b_distribution_net=f'agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed aggregate net of 70 xs 20',
                        reg_p=1,
                        roe=0.10,
                        d2tc=0.3,
                        f_discrete=True,
                        s_values=[.005, 0.01, 0.03],
                        gs_values=[0.029126,   0.047619,   0.074074],
                        bs=1,
                        log2=8,
                        padding=1)
    discrete_eq.full_monty()
    discrete_eq.to_json()
    discrete_eq.browse_exhibits()
