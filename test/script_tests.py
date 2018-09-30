"""
Script tests case for aggregate_project: testing parser and object creation and updating

E.g. to run from Jupyter enter

!python script_test.py -v

"""

import numpy as np
import unittest
import sys

sys.path.insert(0, '/s/telos/python/aggregate_project/')
from aggregate import Aggregate, Severity, Portfolio, Underwriter


class UnderwriterScripts(unittest.TestCase):

    def setUp(self):
        self.uw = Underwriter()

    def test_freq_dists(self):
        test_programs = dict(
            B=('agg B 10 claims sev dhistogram xps [1], [1] binomial 0.2', 10, 1),
            H=('agg H 10 claims sev dhistogram xps [1], [1] poisson', 10, 1),
            I=('agg I 10 claims sev dhistogram xps [1], [1] mixed gamma 0.4', 10, 1),
            J=('agg J 10 claims sev dhistogram xps [1], [1] mixed gamma 1.4', 10, 1),
            K=('agg K 10 claims sev dhistogram xps [1], [1] mixed delaporte 0.4 0.5', 10, 1),
            L=('agg L 10 claims sev dhistogram xps [1], [1] mixed ig 1.4', 10, 1),
            M=('agg M 10 claims sev dhistogram xps [1], [1] mixed sig 0.4 0.5', 10, 1),
            N=('agg N 10 claims sev dhistogram xps [1], [1] mixed pascal 0.4 0.5', 10, 1),
            A=('agg A 2 claims sev 5 * uniform fixed', 10, 0.01),
            E=('agg E 0.2 claims sev 5 * uniform bernoulli', 16, 0.01),
            F=('agg F 10 claims 10 xs 5 sev lognorm 10 cv 2 mixed gamma 0.16', 16, 0.01),
            G=('agg G 10 claims 10 xs 5 sev lognorm 10 cv 2 mixed gamma 1.4', 17, 0.025),
            P=('agg P 10 claims sev lognorm 10 cv 0.765 mixed sig 0.4 0.5', 16, 0.025),
            a=('agg a 1 claim sev dhistogram xps [1 2 3 4] [.25 .25 .25 .25] fixed', 16, 0.025),
            b=('agg b 10 claims sev 5 * uniform binomial 0.2', 16, 0.025),
            c=('agg c 10 claims sev 5 * uniform binomial 0.2', 16, 0.025),
            d=('agg d 10 claims 10 xs 5 sev lognorm 10 cv 2 poisson', 16, 0.025),
            e=('agg e 10 claims 10 xs 5 sev lognorm 10 cv 2 mixed gamma 0.16', 16, 0.025),
            f=('agg f 10 claims 10 xs 5 sev lognorm 10 cv 2 mixed gamma 1.4', 16, 0.05),
            g=('agg g 10 claims sev lognorm 10 cv 0.765 mixed sig 0.4 0.5', 16, 0.025))

        for k, v in test_programs.items():
            self.sc_runner(*v)

    def sc_runner(self, program, log2, bs):
        ag = self.uw(program)
        ag.update(np.arange(1 << log2, dtype=float) * bs)
        # checks
        m, em, cv, ecv = ag.audit_df.loc['mixed', 'agg_m'], ag.audit_df.loc['mixed', 'emp_agg_1'], \
                         ag.audit_df.loc['mixed', 'agg_cv'], ag.audit_df.loc['mixed', 'emp_agg_cv']
        merr, cverr = abs(em / m - 1), abs(ecv / cv - 1)
        self.assertTrue(merr < 1e-5)
        self.assertTrue(cverr < 1e-5)
        return
        # if merr < 1e-5 and cverr < 1e-5:
        #     print(f'OK {program[0:50]:<50s}')
        # else:
        #     print(f'mean err = {merr:.4e}, cverr = {cverr:.4e}, {program[0:50]:<50s}')


if __name__ == '__main__':
    unittest.main()
