
from functools import cache
from itertools import combinations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

mapper = dict(
      L=1 << 0,
      M=1 << 1,
      P=1 << 2,
      Q=1 << 3,
      a=1 << 4,
      lr=1 << 5,
      coc=1 << 6,
      pq=1 << 7
     )

# old method and / or creating the relvant dataframes
def code(r):
    """ determine binary code for row """
    return mapper[r.x] + mapper[r.y] + mapper[r.z]


def proc(s):
    if len(s.intersection(set(['L','M', 'P']))) >= 2:
        s = s.union(set(['L', 'M', 'P', 'lr']))
    if len(s.intersection(set(['P','Q', 'a']))) >= 2:
        s = s.union(set(['P', 'Q', 'a', 'pq']))
    if len(s.intersection(set(['a', 'coc', 'L']))) == 3 or \
        len(s.intersection(set(['a', 'coc', 'lr']))) == 3 or \
        len(s.intersection(set(['L', 'coc', 'pq']))) == 3:
        s = s.union(['P', 'M'])
    for two_of in [set(['L', 'M', 'P','lr']), set(['pq', 'a', 'Q', 'P']),
                        set(['M', 'Q', 'coc'])]:
        if len(s.intersection(two_of)) >= 2:
            s = s.union(two_of)
    return s


def proc4(s):
    return proc(proc(proc(proc(s))))

@cache
def make_possible_pentagons():
    """ enumerate possible and impossible pentagon configurations """
    df = pd.DataFrame(combinations(list('LMPQa') + ['lr', 'coc', 'pq'], 3),
                      columns=['x', 'y', 'z'])

    df['possible'] = True
    for a, r in df.iterrows():
        rs = set(r.iloc[:-1])
        ps = proc4(rs)
        if len(ps) < 8:
            df.loc[a, 'possible'] = False
    df['code'] = df.apply(code, axis=1)
    poss = df.query('possible').copy()
    # notposs = df.query('not possible').copy()
    return poss


class Pentagon():
    # class level variables
    index = ['L', 'P', 'M', 'a', 'Q', 'lr', 'pq', 'coc']

    def __init__(self, obj=None):
        """
        Create Pentagon object, optionally including an :class:`Aggregate` or :class:`Portfolio` object.

        """
        self.obj = obj
        self.L = None
        self.P = None
        self.M = None
        self.a = None
        self.Q = None
        self.lr = None
        self.pq = None
        self.coc = None

    def __str__(self):
        return str(self.as_series())

    def as_series(self):
        """ return values as a pandas Series """
        return pd.Series(self.values,
            index=self.index)

    def as_frame(self):
        """ return values as a pandas DataFrame """
        return self.as_series().to_frame().T

    @property
    def values(self):
        """ return values as a list """
        return [getattr(self, k) for k in self.index]

    def ratios(self):
        """
        Add the ratios (lr, pq, coc) to a partially solved model and / or check they have the expected value
        """
        if self.lr is None:
            self.lr = self.L / self.P
        else:
            assert np.allclose(self.lr, self.L / self.P, atol=1e-14, rtol=1e-14), f'{self.lr} != {self.L / self.P}'
        if self.pq is None:
            self.pq = self.P / self.Q
        else:
            assert np.allclose(self.pq, self.P / self.Q, atol=1e-14, rtol=1e-14), f'{self.pq} != {self.P / self.Q}'
        if self.coc is None:
            self.coc = self.M / self.Q
        else:
            assert np.allclose(self.coc, self.M / self.Q, atol=1e-14, rtol=1e-14), f'{self.coc} != {self.M / self.Q}'

    def solve_obj(self, p, *, P=None, M=None, Q=None, lr=None, pq=None, coc=None):
        """
        Solve using the embedded option to determine a = obj.q(p) and L = obj.density_df.loc[a, 'exa_total']
        Any one of the other variables can be passed in as a keyword argument.
        """
        assert self.obj is not None and self.obj.density_df is not None, \
            'obj must be set and recomputed before calling this method'

        a = self.obj.q(p)
        if 'exa' in self.obj.density_df.columns:
            # Aggregate
            L = self.obj.density_df.loc[a, 'exa']
        elif 'exa_total' in self.obj.density_df.columns:
            # Portfolio
            L = self.obj.density_df.loc[a, 'exa_total']
        else:
            raise ValueError('obj.density_df must contain column exa or exa_total. '
                             'These are provided by Aggregate and Portfolio objects.')
        return self.solve(L=L, P=P, M=M, a=a, Q=Q, lr=lr, pq=pq, coc=coc)

    def solve(self, *, L=None, P=None, M=None, a=None, Q=None, lr=None, pq=None, coc=None):
        """
        Figure the standard variables from a subset for obj, an :class:`Aggregate` or :class:`Portfolio` object.

        """
        # suck out values
        self.L = L
        self.P = P
        self.M = M
        self.a = a
        self.Q = Q
        self.lr = lr
        self.pq = pq
        self.coc = coc

        ser = pd.Series([L, P, M, a, Q, lr, pq, coc],
            index=self.index)

        ser_in = ser[~np.isnan(ser)]

        match tuple(ser_in.index):
            case ('L', 'M', 'Q'):
                self.P = self.L + self.M
                self.a = self.P + self.Q

            case ('L', 'M', 'a'):
                self.P = self.L + self.M
                self.Q = self.a - self.P

            case ('L', 'M', 'coc'):
                self.P = self.L + self.M
                self.Q = self.M / self.coc
                self.a = self.P + self.Q

            case ('L', 'M', 'pq'):
                self.P = self.L + self.M
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('L', 'P', 'Q'):
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case ('L', 'P', 'a'):
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case ('L', 'P', 'coc'):
                self.M = self.P - self.L
                self.Q = self.M / self.coc
                self.a = self.P + self.Q

            case ('L', 'P', 'pq'):
                self.M = self.P - self.L
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('L', 'a', 'Q'):
                self.P = self.a - self.Q
                self.M = self.P - self.L

            case ('L', 'Q', 'lr'):
                self.P = self.L / self.lr
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case ('L', 'Q', 'coc'):
                self.M = self.Q * self.coc
                self.P = self.L + self.M
                self.a = self.P + self.Q

            case ('L', 'Q', 'pq'):
                self.P = self.Q * self.pq
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case ('L', 'a', 'lr'):
                self.P = self.L / self.lr
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case ('L', 'a', 'coc'):
                self.M = self.coc / (1 + self.coc) * (self.a - self.L)
                self.P = self.L + self.M
                self.Q = self.a - self.P

            case ('L', 'a', 'pq'):
                self.P = self.a * self.pq / (1 + self.pq)
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case ('L', 'lr', 'coc'):
                self.P = self.L / self.lr
                self.M = self.P - self.L
                self.Q = self.M / self.coc
                self.a = self.P + self.Q

            case ('L', 'lr', 'pq'):
                self.P = self.L / self.lr
                self.M = self.P - self.L
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('L', 'pq', 'coc'):
                self.P = self.pq / (self.pq - self.coc) * self.L
                self.M = self.P - self.L
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('P', 'M', 'Q'):
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case ('P', 'M', 'a'):
                self.L = self.P - self.M
                self.Q = self.a - self.P

            case ('P', 'M', 'coc'):
                self.L = self.P - self.M
                self.Q = self.M / self.coc
                self.a = self.P + self.Q

            case ('P', 'M', 'pq'):
                self.L = self.P - self.M
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('M', 'a', 'Q'):
                self.P = self.a - self.Q
                self.L = self.P - self.M
                self.M = self.P - self.L

            case ('M', 'Q', 'lr'):
                self.L = self.M  * self.lr / (1 - self.lr)
                self.P = self.L + self.M
                self.a = self.P + self.Q

            case ('M', 'Q', 'pq'):
                self.P = self.Q * self.pq
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case ('M', 'a', 'lr'):
                self.P = self.M  / (1 - self.lr)
                self.L = self.P - self.M
                self.Q = self.a - self.P

            case ('M', 'a', 'coc'):
                self.Q = self.M / self.coc
                self.P = self.a - self.Q
                self.L = self.P - self.M

            case ('M', 'a', 'pq'):
                self.P = self.a * self.pq / (1 + self.pq)
                self.L = self.P - self.M
                self.Q = self.a - self.P

            case ('M', 'lr', 'coc'):
                self.P = self.M / (1 - self.lr)
                self.L = self.P - self.M
                self.Q = self.M / self.coc
                self.a = self.P + self.Q

            case ('M', 'lr', 'pq'):
                self.P = self.M / (1 - self.lr)
                self.L = self.P - self.M
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('M', 'pq', 'coc'):
                self.P = self.pq / self.coc * self.M
                self.L = self.P - self.M
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('P', 'Q', 'lr'):
                self.L = self.P * self.lr
                self.a = self.P + self.Q
                self.M = self.P - self.L

            case ('P', 'Q', 'coc'):
                self.M = self.Q * self.coc
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case ('P', 'a', 'lr'):
                self.L = self.P * self.lr
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case ('P', 'a', 'coc'):
                self.Q = self.a - self.P
                self.M = self.Q * self.coc
                self.L = self.P - self.M

            case ('P', 'lr', 'coc'):
                self.L = self.P * self.lr
                self.M = self.P - self.L
                self.Q = self.M / self.coc
                self.a = self.P + self.Q

            case ('P', 'lr', 'pq'):
                self.L = self.P * self.lr
                self.M = self.P - self.L
                self.Q = self.P / self.pq
                self.a = self.P + self.Q

            case ('P', 'pq', 'coc'):
                self.Q = self.P / self.pq
                self.a = self.P + self.Q
                self.M = self.Q * self.coc
                self.L = self.P - self.M

            case ('a', 'Q', 'lr'):
                self.P = self.a - self.Q
                self.L = self.P * self.lr
                self.M = self.P - self.L

            case ('a', 'Q', 'coc'):
                self.P = self.a - self.Q
                self.M = self.Q * self.coc
                self.L = self.P - self.M

            case ('Q', 'lr', 'coc'):
                self.M = self.Q * self.coc
                self.P = self.M / (1 - self.lr)
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case ('Q', 'lr', 'pq'):
                self.P = self.Q * self.pq
                self.L = self.P * self.lr
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case ('Q', 'pq', 'coc'):
                self.P = self.Q * self.pq
                self.M = self.Q * self.coc
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case ('a', 'lr', 'coc'):
                self.P = self.a * self.coc / (self.coc + 1 - self.lr)
                self.Q = self.a - self.P
                self.L = self.P * self.lr
                self.M = self.P - self.L

            case ('a', 'lr', 'pq'):
                self.P = self.pq * self.a / (1 + self.pq)
                self.L = self.P * self.lr
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case ('a', 'pq', 'coc'):
                self.P = self.pq * self.a / (1 + self.pq)
                self.Q = self.a - self.P
                self.M = self.Q * self.coc
                self.L = self.P - self.M

            case _:
                raise ValueError(f'Insoluble case: {tuple(ser_in.index)}')

        # fill in the ratios
        self.ratios()
        return self.as_frame()

    @classmethod
    def test_cases(cls, L, P, a):
        """
        Run all test cases for a given set of loss, premium, and asset inputs.

        """
        M = P - L
        Q = a - P
        lr = L / P
        coc = M / Q
        pq = P / Q
        consistent_pricing = pd.Series(
            # TODO fragile...must be in the same order as cls.index
            [L, P, M, a, Q, lr, pq, coc],
            index=cls.index
            )

        # run through all options
        p = Pentagon()
        df = make_possible_pentagons()
        # add columns for each variable
        for c in cls.index:
            df[c] = None

        # iterate through all possible combinations
        for i, r in df.iterrows():
            arg_dict = {k: consistent_pricing[k] for k in r.iloc[:3]}
            p.solve(**arg_dict)
            # print(p.values)
            df.loc[i, cls.index] = p.values

        return df
