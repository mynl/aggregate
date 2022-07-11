# scratch file

import sys
import aggregate as agg
from aggregate import build
from aggregate.utils import make_ceder_netter
from examples import case_studies as cs

from numpy import exp
from scipy.interpolate import interp1d

# ISSUES =================================================================
# The million issues with matplotlib fonts!!!!
# don't forget about this: df, ans = agg.frequency_examples(n=100, ν=0.45, f=0.5, κ=1.25, sichel_case='', log2=16, xmax=2500)
# or this: ans = net.entropy_fit(4)

# net occ with high claim count and moment matching: works?

# print(os.environ['PYTHONPATH'].split(';'))

# a sensible level
# logger_level(30)

section = []
if len(sys.argv) > 1:
    # there are args
    if sys.argv[1] == 'h' or sys.argv[1] == '-h':
        print('Select options from:\n    parse_tests\n    scratch\n    scratch_ex\n    parse_test\n    netters_and_ceders\n    simple_discrete\n    hu_scs_case\n    sev_intro\n    easter_egg.')
        print('\nSelect no options just to import build etc.')
    else:
        section = sys.argv[1:]


# section = [
    # 'scratch',
    # 'book_case_studies',
    # 'parse_tests',
    # 'netters_and_ceders',
    # 'simple_discrete',
    # 'sev_intro'
    # 'easter_egg'
# ]


def scratch_XX():
    pass


if 'scratch' in section:
    # do some parser testing!
    print(
        f'ACTIVATING scratch SECTION {__name__} ===========================')

    # on logger 10 this produces the long list... on first run only
    gross = build(
        'agg TestA 1 claim sev lognorm 30 cv .3 net of 60 x 40 occurrence fixed ')
    net = build(
        'agg TestB 2 claim sev lognorm 30 cv .3 fixed net of 60 x 40 aggregate')

    net.plot()
    display(net)

    # use -v option to translate
    f, axs = smfig(1, 1, (5.0, 3.0), )
    ax = ax0 = axs
    axs = np.array([axs])
    ax.plot(gross.xs, gross.sev_net_density.cumsum(), label='net')
    ax.plot(gross.xs, gross.sev_gross_density.cumsum(), ls=":", label='gross')
    ax.plot(gross.xs, gross.sev_ceded_density.cumsum(), ls='--', label='ceded')
    # ax.set(ylim=b.limits('density', 'linear', 'exclude'))
    ax.legend(loc='lower right')

    f, axs = smfig(1, 1, (5.0, 3.0), )
    ax = ax0 = axs
    axs = np.array([axs])
    ax.plot(net.xs, net.agg_net_density.cumsum(), label='net')
    ax.plot(net.xs, net.agg_gross_density.cumsum(), ls=":", label='gross')
    ax.plot(net.xs, net.agg_ceded_density.cumsum(), ls='--', label='ceded')
    # ax.set(ylim=a.limits('density', 'linear', 'exclude'))
    ax.legend(loc='lower right')


# ==========================================================================================


def book_case_studies_XX():
    pass


if 'book_case_studies' in section:
    print(
        f'ACTIVATING book_case_studies SECTION {__name__} ===========================')

    discrete = cs.CaseStudy()
    discrete.factory(case_id='discrete_new',
                     case_name='Discrete Case',
                     case_description='Discrete Case in the new syntax.',
                     a_distribution='agg X1 1 claim dsev [0 8 10] [1/2 1/4 1/4] fixed',
                     b_distribution_gross='agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed',
                     b_distribution_net=f'agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed aggregate net of 70 xs 20',
                     reg_p=1,
                     roe=0.10,
                     d2tc=0.3,
                     f_discrete=True,
                     f_blend_extend=True,
                     bs=1,
                     log2=8,
                     padding=1)
    discrete.full_monty()
    discrete.to_json()

    discrete_eq = cs.CaseStudy()
    discrete_eq.factory(case_id='discrete_equal_new',
                       case_name='Discrete Case, equal points',
                       case_description='Discrete Case, equal points, in the new syntax.',
                       a_distribution       = 'agg X1 1 claim dsev [0 9 10] [1/2 1/4 1/4] fixed',
                       b_distribution_gross = 'agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed',
                       b_distribution_net   =f'agg X2 1 claim dsev [0 1 90] [1/2 1/4 1/4] fixed aggregate net of 70 xs 20',
                       reg_p=1,
                       roe=0.10,
                       d2tc=0.3,
                       f_discrete=True,
                       f_blend_extend=True,
                       bs=1,
                       log2=8,
                       padding=1)
    discrete_eq.full_monty()
    discrete_eq.to_json()


    # tame
    recalc = build('agg B 1 claim sev gamma  50 cv 0.15 fixed',
                   log2=16, bs=1/64)
    a, d = recalc.q(0.8), recalc.q(0.99)
    y = d - a
    tame = cs.CaseStudy()
    tame.factory(case_id='tame_new',
                 case_name='Tame Case',
                 case_description='Tame Case in the new syntax.',
                 a_distribution='agg A 1 claim sev gamma  50 cv 0.10 fixed',
                 b_distribution_gross='agg B 1 claim sev gamma  50 cv 0.15 fixed',
                 b_distribution_net=f'agg B 1 claim sev gamma  50 cv 0.15 fixed aggregate net of {y} xs {a}',
                 reg_p=0.999,
                 roe=0.10,
                 d2tc=0.3,
                 f_discrete=False,
                 f_blend_extend=True,
                 bs=1/64,
                 log2=16,
                 padding=1)
    tame.full_monty()
    tame.to_json()

    # CNC
    recalc = build(
        'agg Cat    1 claim sev lognorm  20 cv 1.00 fixed', log2=16, bs=1/64)
    a, d = recalc.q(0.9), recalc.q(0.995)
    y = d - a
    cnc = cs.CaseStudy()
    cnc.factory(case_id='cnc_new',
                case_name='Cat/Non-Cat',
                case_description='Cat/Non-Cat in the new syntax.',
                a_distribution='agg NonCat 1 claim sev gamma    80 cv 0.15 fixed',
                b_distribution_gross='agg Cat 1 claim sev lognorm  20 cv 1.00 fixed',
                b_distribution_net=f'agg Cat 1 claim sev lognorm  20 cv 1.00 fixed aggregate net of {y} xs {a}',
                reg_p=0.999,
                roe=0.10,
                d2tc=0.3,
                f_discrete=False,
                f_blend_extend=True,
                bs=1/64,
                log2=16,
                padding=1)
    cnc.full_monty()
    cnc.to_json()

    # hs
    # these are numbers from the new_case_studies md files
    freq1, sigma1 = 70, 1.9
    freq2, sigma2 = 2, 2.5
    mu1 = np.log(70 / freq1) - sigma1**2 / 2
    mu2 = np.log(30 / freq2) - sigma2**2 / 2
    print(freq1, exp(mu1), sigma1, freq2, exp(
        mu2), sigma2, mu1, sigma1, mu2, sigma2)

    recalc = build('sev Husev  15 * exp(0 - 2.5**2 / 2) @ lognorm 2.5')
    a, d = recalc.isf(0.05), recalc.isf(0.005)
    y = d - a
    hs = cs.CaseStudy()
    hs.factory(case_id='hs_new',
                       case_name='Hu/SCS Case',
                       case_description='Hu/SCS Case in the new syntax.',
                       a_distribution='agg SCS 70 claims sev exp(0 - 1.9**2 / 2)      @ lognorm 1.9 poisson',
                       b_distribution_gross='agg Hu  2 claims sev 15 * exp(0 - 2.5**2 / 2) @ lognorm 2.5 poisson',
                       b_distribution_net=f'agg Hu   2 claims sev 15 * exp(0 - 2.5**2 / 2) @ lognorm 2.5 poisson aggregate net of {y} xs {a}',
                       reg_p=0.999,
                       roe=0.10,
                       d2tc=0.3,
                       f_discrete=False,
                       f_blend_extend=True,
                       bs=1/4,
                       log2=19,
                       padding=1)
    hs.full_monty()
    hs.to_json()

    hs2 = cs.CaseStudy()
    hs2.factory(case_id='hs_per_occ',
                case_name='Hu/SCS',
                case_description='Hu/SCS Case in the new syntax with per occurrence reinsurance .',
                a_distribution='agg SCS 70 claims sev exp(0 - 1.9**2 / 2)      @ lognorm 1.9 poisson',
                b_distribution_gross='agg Hu   2 claims sev 15 * exp(0 - 2.5**2 / 2) @ lognorm 2.5 poisson',
                b_distribution_net=f'agg Hu   2 claims sev 15 * exp(0 - 2.5**2 / 2) @ lognorm 2.5 occurrence net of {y} xs {a} poisson',
                reg_p=0.999,
                roe=0.10,
                d2tc=0.3,
                f_discrete=False,
                f_blend_extend=True,
                bs=1/4,
                log2=19,
                padding=1)
    hs2.full_monty()
    hs2.to_json()


def parse_tests_XX():
    pass


if 'parse_tests' in section:
    print(
        f'ACTIVATING parse_tests SECTION {__name__} ===========================')

    # do some parser testing!
    build('sev One dhistogram xps [1] [1]')
    build('sev ONE dsev [1]')

    df = build.interpreter_file(where='')
    display(df)

    tests = [
        'agg A 1 claim dsev [1 2 3] fixed',
        'agg B 2 claim dsev [1 2 3] fixed',
        'agg C 1 claim dsev [1 2 3 10] fixed',
        'sev S dsev[1 2 3 5 10 25 100]',
        'sev T dsev[1 2 3 5 10 25 100 1000]',
        'port PORT0\n\tagg XLN 1 claim sev lognorm 20 cv .9 poisson',
        'port PORTA\n\tagg.A',
        'port PORTB\n\tagg.A\n\tagg.B',
        'port PORTC\n\tagg NewA 3 @ agg.A\n\tagg.C',
        'port PORTD\n\tagg AA 1 claim sev.S fixed\n\tagg AB 1 claim sev sev.T fixed',
        'port PORTE\n\tagg S2    2 claims sev 3 @ sev.S fixed\n\tagg T~5   2 claims sev sev.T # 5 fixed\n\tagg T_4_5 2 claims sev 4 @ sev.T # 5 fixed',
        'port PORTF\n\tagg S1  3 * agg.A\n\tagg S2  4 @ agg.A\n\tagg S3  agg.A # 6\n\tagg S4   9 * 3 @ agg.A # 6'
    ]
    # for parse to work actually have to build referents
    build('agg A 1 claim dsev [1 2 3] fixed')
    build('agg B 2 claim dsev [1 2 3] fixed')
    build('agg C 1 claim dsev [1 2 3 10] fixed')
    build('sev S dsev[1 2 3 5 10 25 100]')
    build('sev T dsev[1 2 3 5 10 25 100 1000]')
    display(build.knowledge)

    display(build.interpreter_list(tests))

    # for test in tests:
    #     display(build.interpreter_one(test))

    ans = {}
    build.uw.create_all = False
    for n, test in enumerate(tests):
        ans[n] = build(test)
        display(ans[n])
    build.uw.create_all = True

# ==========================================================================================


def netters_and_ceders_XX():
    pass


if 'netters_and_ceders' in section:
    # do some parser testing!
    print(
        f'ACTIVATING netters_and_ceders SECTION {__name__} ====================')

    a = build(
        'agg Gross   1 claim sev lognorm 30 cv .3                             fixed ')
    b = build(
        'agg NetOf   1 claim sev lognorm 30 cv .3 net of   60 x 40 occurrence fixed ')
    c = build(
        'agg CededTo 1 claim sev lognorm 30 cv .3 ceded to 60 x 40 occurrence fixed')

    if 1:
        a = build('agg TestA 15 claims sev lognorm 30 cv .3 poisson')
        a.plot()
        a.report()
        a
        aa = build(
            'agg TestA1 1 claim [1 2 5 10] xs 0 sev lognorm .5 cv 1 fixed')
        aa.plot()
        aa.report()
        aa
        b = build('agg TestB 15 claims sev constant 1 poisson')
        b.plot()
        b.report()
        b

        c = build(
            'agg TestC specified claims sev constant 1 nps [0 1 2 10] [.3 .2 .25 .25]')
        c.plot()
        c.report()
        c

        # make something to work with
        gross = build(
            'agg OccReEgGross 5 claim 100 x 0 sev lognorm 20 cv 1.25 poisson')
        gross.plot()
        display(gross.report())  # little more detail than describe
        # gross.describe()  # display is used by repr_html
        display(gross)

        net = build('agg OccReEgNet 5 claim 100 x 0 sev lognorm 20 cv 1.25 net of 50% so 5 x 5 and 10 x 10 and .95 so 30 x 20 and 25 po 50 x 50 occurrence poisson', debug=True)
        net.plot()
        display(net)

        net.reins_audit_df

        # more of a cat/umbrella/comm auto example
        gross = build(
            'agg OccReEgGross 10 claims 10000 x 0 sev lognorm 40 cv 1. poisson')
        gross.plot()
        display(gross)

        net = build('agg OccReEgNet 10 claims 10000 x 0 sev lognorm 40 cv 1. net of 50% so 1 x 1 and .95 so 3 x 2 and 4 po 5 x 5 occurrence poisson', debug=True)
        net.plot()
        display(net)

        p = 0.999
        g = agg.Distortion('wang', 0.5)
        pd.concat((gross.price(0.999, g), net.price(p, g)))

# ==========================================================================================


def simple_discrete_XX():
    pass


if 'simple_discrete' in section:
    a = build(
        'agg A 1 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] fixed')
    display(a.spec)
    display(a.density_df)
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot()

    ans = {}
    for n in range(1, 10):
        ans[n] = build(
            f'agg A{n} {n} claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] fixed')
        display(ans[n])
        ans[n].density_df.query('p_total > 0').p_total.plot(
            ax=ax, ls='-', marker='o', c=f'C{n % 12}', label=f'n={n}')

    ax.legend()

    a
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot()

    ans = {}
    for n in range(1, 10):
        ans[n] = build(
            f'agg A{n} {n} claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] fixed')
        # display(ans[n])
        ans[n].density_df.query('p_total > 0').p_total.plot(
            ax=ax, ls='-', marker='o', c=f'C{n % 12}', label=f'n={n}')

    ax.legend()
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot()

    ans = {}
    for n in range(1, 10):
        ans[n] = build(
            f'agg A{n} {n} claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] fixed')
        # display(ans[n])
        ans[n].density_df.query('p_total > 0').p_total.plot(
            ax=ax, ls='-', marker='o', c=f'C{n % 12}', label=f'n={n}')

    ax.legend()
    ax.grid(lw=.25, c='w')
    a.density_df.query('p_total > 0').p_total.plot(ls='-', marker='o')
    a.density_df.query('p_total > -10').p_total.plot(ls='-', marker='o')
    a.density_df.query('p_total > -10').p_total.plot(ls='-',
                                                     marker='o', drawstyle='steps-post')
    a.density_df.p_total.plot(ls='-', marker='o', drawstyle='steps-post')
    a.density_df.query('p_total > 0').p_total.plot(
        ls='-', marker='o', drawstyle='steps-post')
    9*6, (1/6)**9
    9*6, (1/6)**9, ans[9].density_df.loc[54, 'p_total']
    9*6, (1/6)**9, ans[9].density_df.loc[54, 'p_total'], 1 - \
        ans[9].density_df.loc[54, 'p_total'] / (1/6)**9
    build.uw
    build.uw.last_spec
    uw = build.uw
    for x in [uw['A9'],
              ans[9],
              ans[9].spec_ex,
              ans[9].report(),
              ans[9].statistics_df,
              ans[9].report_ser,
              ans[9].statistics_total_df.T,
              ans[9].statistics_df.T]:
        display(x)

    ans[9].plot()
    ans[9].plot(figsize=(12, 4))
    print(ans[9].program)
    uw.describe()
    uw.list()
    uw.describe()
    k, s = uw['A9']
    k
    s
    mks = agg.Aggregate(**s)
    mks
    mks.easy_update(log2=5)
    mks

    uw.parse_portfolio_program(
        'agg A 1 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] fixed')
    uw.parse_portfolio_program(
        'agg A 1 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] fixed net of 2 x 2 aggregate')
    uw.parse_portfolio_program(
        'agg A 1 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] net of 0.5 so 3 x 3 occurrence fixed net of 2 x 2 aggregate')
    uw.parse_portfolio_program(
        'agg A 10 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] net of 0.5 so 3 x 3 occurrence mixed gamma 0.4 net of 2 x 2 aggregate')
    uw.parse_portfolio_program(
        'agg A 10 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] net of 0.5 so 3 x 3 occurrence mixed gamma 0.4 net of 2 x 2 aggregate // comment')
    uw.parse_portfolio_program(
        'agg A 10 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] net of 0.5 so 3 x 3 occurrence mixed gamma 0.4 net of 2 x 2 aggregate /\n note{some}')
    uw.parse_portfolio_program(
        'agg A 10 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] net of 0.5 so 3 x 3 occurrence mixed gamma 0.4 net of 2 x 2 aggregate note{some}')
    uw.parse_portfolio_program(
        'agg A 10 claims sev dhistogram xps [1 2 3 4 5 6] [1/6 1/6 1/6 1/6 1/6 1/6] net of 0.5 so 3 x 3 occurrence mixed gamma 0.4 net of 2 x 2 aggregate \\\n note{some}')


# ==========================================================================================


def sev_intro_XX():
    pass


if 'sev_intro' in section:
    # do some parser testing!
    print(
        f'ACTIVATING sev_intro SECTION {__name__} ===========================')

    s = build('sev sevA lognorm 20 cv .3')

    s.plot()

