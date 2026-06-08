"""Pentagon — the accounting authority for the L–M–P–Q–a pricing identities.

Every pricing readout in the library emits the same eight accounting
quantities: the five *amounts*

* ``L`` — expected loss,
* ``M`` — margin,
* ``P`` — premium (``P = L + M``),
* ``Q`` — capital / equity,
* ``a`` — assets (``a = P + Q``),

and the three *ratios*

* ``LR = L / P`` — loss ratio,
* ``PQ = P / Q`` — premium-to-surplus,
* ``ROE = M / Q`` — return on equity (a.k.a. cost of capital, ``CoC``).

This module is the single source of truth for

1. the **canonical names and order** (:data:`PENTAGON_STATS`,
   :data:`PENTAGON_DTYPE`) used by every emitter, so readouts concatenate and
   diff cleanly;
2. the **completion identities** — :func:`complete_pentagon` (vectorized,
   the common "I already hold ``L, M, P, Q``" path) and :class:`Pentagon`
   (:meth:`Pentagon.solve`, the single-row engine that completes *any* soluble
   triple, e.g. ``P, L`` and ``a`` *or* ``Q``).

The vectorized helper and the single-row object express the *same* identities
two ways; emitters that produce frames complete their rows through
:func:`complete_pentagon`, while a caller holding a partial spec reaches for
:class:`Pentagon`.

Per ``CLAUDE.md`` this stays submodule-access only (no top-level re-export);
it is an internal contract, not public surface. It is a numpy/pandas-only leaf,
so ``portfolio`` and ``distributions`` import it with no cycle.

Notes
-----
``ROE`` vs ``CoC``: the same number ``M / Q`` is read two ways. ``ROE`` is the
*ex post* view (the return achieved on equity, given a P&L presentation);
``CoC`` is the *prospective* pricing view (the capital charge baked into the
premium). The library presents pricing as a P&L, so the emitted column is
named ``ROE`` with ``CoC`` documented as the synonym. (For an ``Aggregate`` or
``Portfolio`` all capital is equity — the capital-vs-equity distinction only
arises under capital tranching, which is out of scope here.)
"""

from collections import namedtuple
from functools import cache
from itertools import combinations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The canonical contract
# ---------------------------------------------------------------------------
#: Canonical stat names in accounting reading order: amounts (``L + M = P``,
#: then ``P + Q = a``) followed by the three ratios. Used as the stat axis of
#: every pricing readout in the library.
PENTAGON_STATS = ['L', 'M', 'P', 'Q', 'a', 'LR', 'PQ', 'ROE']

#: Ordered ``CategoricalDtype`` over :data:`PENTAGON_STATS`, so a stat axis
#: sorts canonically and survives round-trips.
PENTAGON_DTYPE = pd.CategoricalDtype(categories=PENTAGON_STATS, ordered=True)

#: The five amount stats.
CORE_STATS = ['L', 'M', 'P', 'Q', 'a']

#: The three ratio stats.
RATIO_STATS = ['LR', 'PQ', 'ROE']


def complete_pentagon(df, *, stat_dtype=True):
    """Fill ``a`` + ratios and return canonical, octet-trailing columns.

    Vectorized completion for the common case where every row already carries
    the core amounts ``L``, ``M``, ``P`` and ``Q`` (the emitters all build
    these from an augmented-distortion row). Computes ``a = P + Q`` and the
    three ratios, then reorders so that any *descriptor* columns (anything not
    in :data:`PENTAGON_STATS`) come first and the eight pentagon stats are the
    trailing block — extractable with a fixed ``df.iloc[:, -8:]`` regardless of
    how many descriptors precede them.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``L``, ``M``, ``P``, ``Q``. May contain any
        number of additional descriptor columns (e.g. ``dname``, ``dshape``).
    stat_dtype : bool, default True
        When the frame has no descriptor columns (its columns are exactly the
        eight stats), stamp the column index with :data:`PENTAGON_DTYPE`
        (named ``'stat'``). Skipped when descriptors are present, since a mixed
        descriptor/stat axis cannot be a single categorical.

    Returns
    -------
    pandas.DataFrame
        Same rows; columns are ``[*descriptors, *PENTAGON_STATS]``.

    Notes
    -----
    The identities are ``a = P + Q``, ``LR = L / P``, ``PQ = P / Q`` and
    ``ROE = M / Q``. This is the one implementation that every frame-emitter
    routes through, replacing the previously hand-copied derivation blocks.
    """
    out = df.copy()
    out['a'] = out['P'] + out['Q']
    out['LR'] = out['L'] / out['P']
    out['PQ'] = out['P'] / out['Q']
    out['ROE'] = out['M'] / out['Q']
    descriptors = [c for c in out.columns if c not in PENTAGON_STATS]
    out = out[descriptors + PENTAGON_STATS]
    if stat_dtype and not descriptors:
        out.columns = pd.CategoricalIndex(
            out.columns, dtype=PENTAGON_DTYPE, name='stat')
    return out


def apply_pentagon_columns(df, *, axis='columns'):
    """Stamp the canonical categorical on a stat axis in canonical order.

    Use when an axis already holds exactly the eight stat labels (possibly in
    the wrong order, or as plain strings after a transpose that dropped the
    categorical dtype). Reindexes the chosen axis to :data:`PENTAGON_STATS`
    and applies :data:`PENTAGON_DTYPE`.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame whose ``axis`` is the eight pentagon stats.
    axis : {'columns', 'index'}, default 'columns'
        Which axis carries the stats.

    Returns
    -------
    pandas.DataFrame
        Reordered with the stat axis as an ordered ``CategoricalIndex`` named
        ``'stat'``.
    """
    if axis == 'columns':
        out = df.reindex(columns=PENTAGON_STATS)
        out.columns = pd.CategoricalIndex(
            out.columns, dtype=PENTAGON_DTYPE, name='stat')
    elif axis == 'index':
        out = df.reindex(index=PENTAGON_STATS)
        out.index = pd.CategoricalIndex(
            out.index, dtype=PENTAGON_DTYPE, name='stat')
    else:
        raise ValueError(f"axis must be 'columns' or 'index', got {axis!r}")
    return out


# ---------------------------------------------------------------------------
# enumeration of soluble configurations (used by Pentagon.test_cases)
# ---------------------------------------------------------------------------
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

pent_ans = namedtuple('pent_ans', 'L M P Q a LR PQ ROE')

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
    """Single-row pricing record (an eight-vector) and completion engine.

    Holds the eight pentagon quantities as named attributes (``L``, ``M``,
    ``P``, ``Q``, ``a``, ``LR``, ``PQ``, ``ROE``) and completes a partial
    specification via :meth:`solve` — given any soluble triple it fills the
    rest with consistency checks. Optional provenance (a ``distortion`` object
    and its ``shape``) rides as attributes, never as data rows.

    This is the single-row analogue of :func:`complete_pentagon`; both express
    the same accounting identities. The object is *not* a ``DataFrame`` (pandas
    subclassing is brittle) — it *emits* canonical pandas output via
    :meth:`as_series` / :meth:`as_frame`.

    Parameters
    ----------
    obj : Aggregate or Portfolio, optional
        Source object, enabling :meth:`solve_obj` to read ``L`` and ``a`` off
        the embedded ``density_df`` at a probability level.

    Attributes
    ----------
    distortion : Distortion or None
        Optional provenance: the distortion this record was priced under.
    shape : float or None
        Optional provenance: the distortion's shape parameter.
    """
    # canonical stat order; matches PENTAGON_STATS
    index = list(PENTAGON_STATS)

    def __init__(self, obj=None):
        self.obj = obj
        self.L = None
        self.M = None
        self.P = None
        self.Q = None
        self.a = None
        self.LR = None
        self.PQ = None
        self.ROE = None
        self.distortion = None
        self.shape = None

    def __str__(self):
        return str(self.as_series())

    def as_tuple(self):
        """Return the eight values as a named tuple (canonical order)."""
        return pent_ans(*self.values)

    def as_series(self):
        """Return the eight values as a ``pandas.Series`` (canonical order)."""
        return pd.Series(self.values, index=pd.Index(self.index, name='stat'))

    def as_frame(self, line='total'):
        """Return a canonical one-row ``DataFrame``.

        Row indexed by ``line`` (name ``'line'``); columns are the eight
        :data:`PENTAGON_STATS` with :data:`PENTAGON_DTYPE`.

        Parameters
        ----------
        line : str, default 'total'
            Index label for the single row.
        """
        df = pd.DataFrame([self.values], index=pd.Index([line], name='line'),
                          columns=self.index)
        df.columns = pd.CategoricalIndex(
            df.columns, dtype=PENTAGON_DTYPE, name='stat')
        return df

    @property
    def values(self):
        """The eight values as a list, in canonical :attr:`index` order."""
        return [getattr(self, k) for k in self.index]

    def ratios(self):
        """Fill ``LR``, ``PQ``, ``ROE`` from the amounts, or check consistency.

        For each ratio: set it from the amounts if missing, otherwise assert it
        agrees with ``L/P`` / ``P/Q`` / ``M/Q`` to within ``1e-14``.
        """
        if self.LR is None:
            self.LR = self.L / self.P
        else:
            assert np.allclose(self.LR, self.L / self.P, atol=1e-14, rtol=1e-14), f'{self.LR} != {self.L / self.P}'
        if self.PQ is None:
            self.PQ = self.P / self.Q
        else:
            assert np.allclose(self.PQ, self.P / self.Q, atol=1e-14, rtol=1e-14), f'{self.PQ} != {self.P / self.Q}'
        if self.ROE is None:
            self.ROE = self.M / self.Q
        else:
            assert np.allclose(self.ROE, self.M / self.Q, atol=1e-14, rtol=1e-14), f'{self.ROE} != {self.M / self.Q}'

    @classmethod
    def from_row(cls, row, line='total', *, distortion=None, obj=None):
        """Build a solved ``Pentagon`` from an augmented-distortion row.

        Pulls the core amounts for ``line`` off a single row of an
        ``apply_distortion`` augmented frame (``exa_{line}``/``exag_{line}``/
        ``T.M_{line}``/``T.Q_{line}``; the unsuffixed ``exa``/``exag`` for a
        bare ``Aggregate``), solves, and attaches optional provenance.

        Parameters
        ----------
        row : pandas.Series
            One row of an augmented ``density_df`` (e.g. ``aug.loc[a_reg]``).
        line : str, default 'total'
            Which unit to read. ``'total'`` for the portfolio total.
        distortion : Distortion, optional
            Provenance, stored on ``.distortion`` (and ``.shape``).
        obj : Aggregate or Portfolio, optional
            Source object, stored on ``.obj``.

        Returns
        -------
        Pentagon
            Fully solved (all eight stats populated).

        Notes
        -----
        A bare ``Aggregate`` augmented row exposes ``exa``/``exag`` without a
        line suffix; a ``Portfolio`` exposes ``exa_{line}`` etc. This reads
        whichever is present.
        """
        p = cls(obj=obj)
        if f'exa_{line}' in row.index:
            L = row[f'exa_{line}']
            P = row[f'exag_{line}']
            M = row[f'T.M_{line}']
            Q = row[f'T.Q_{line}']
        else:
            # bare Aggregate row
            L = row['exa']
            P = row['exag']
            M = P - L
            Q = None  # not available without an asset level; solve from L,P,M
        if Q is None:
            p.solve(L=L, P=P, M=M)
        else:
            p.solve(L=L, P=P, M=M, Q=Q)
        if distortion is not None:
            p.distortion = distortion
            p.shape = getattr(distortion, 'shape', None)
        return p

    def solve_obj(self, *, p=None, a=None, P=None, M=None, Q=None, lr=None, pq=None, roe=None):
        """Solve the pentagon at a capital level read off the embedded object.

        Fix the capital level with exactly one of ``p`` (a VaR/quantile
        probability, ``a = obj.q(p)``) or ``a`` (an asset level, snapped to the
        ``density_df`` grid). The expected loss ``L`` is read from the object's
        ``exa`` (Aggregate) or ``exa_total`` (Portfolio) column at that level.
        Pass one further target among ``P, M, Q, lr, pq, roe``; the resulting
        triple ``{L, a, target}`` is handed to :meth:`solve`.

        Parameters
        ----------
        p : float, optional
            VaR probability; mutually exclusive with ``a``.
        a : float, optional
            Asset level, snapped to the grid; mutually exclusive with ``p``.
        P, M, Q, lr, pq, roe : float, optional
            One pricing target (see :meth:`solve`).
        """
        assert self.obj is not None and self.obj.density_df is not None, \
            'obj must be set and recomputed before calling this method'
        if (p is None) == (a is None):
            raise ValueError('pass exactly one of p= or a=')

        a = self.obj.q(p) if a is None else self.obj.snap(a)
        if 'exa' in self.obj.density_df.columns:
            # Aggregate
            L = self.obj.density_df.loc[a, 'exa']
        elif 'exa_total' in self.obj.density_df.columns:
            # Portfolio
            L = self.obj.density_df.loc[a, 'exa_total']
        else:
            raise ValueError('obj.density_df must contain column exa or exa_total. '
                             'These are provided by Aggregate and Portfolio objects.')
        return self.solve(L=L, P=P, M=M, a=a, Q=Q, lr=lr, pq=pq, roe=roe)

    def solve(self, *, L=None, P=None, M=None, a=None, Q=None, lr=None, pq=None, roe=None):
        """Figure all eight quantities from any soluble triple.

        Pass exactly three of ``L, P, M, a, Q`` (amounts) and/or ``lr, pq, roe``
        (ratios). The :data:`make_possible_pentagons` enumeration lists the
        soluble triples; an insoluble one raises ``ValueError``.

        Parameters
        ----------
        L, P, M, a, Q : float, optional
            Amount inputs (loss, premium, margin, assets, capital).
        lr, pq, roe : float, optional
            Ratio inputs (``L/P``, ``P/Q``, ``M/Q``). ``roe`` is ``M/Q``,
            the cost of capital.

        Notes
        -----
        Lowercase ``lr``/``pq``/``roe`` are accepted as the keyword spelling of
        the ratios; the solved values land on the uppercase ``LR``/``PQ``/
        ``ROE`` attributes (the canonical stat names).
        """
        # suck out values onto the canonical (uppercase) attributes
        self.L = L
        self.M = M
        self.P = P
        self.Q = Q
        self.a = a
        self.LR = lr
        self.PQ = pq
        self.ROE = roe

        # match key uses the ratio spellings lr/pq/coc for the enumeration
        ser = pd.Series([L, M, P, Q, a, lr, pq, roe],
            index=['L', 'M', 'P', 'Q', 'a', 'lr', 'pq', 'coc'])

        ser_in = ser[~pd.isna(ser)]

        # the match is keyed on a frozenset of the supplied names so input
        # order is irrelevant; canonicalize to the sorted-by-name tuple.
        supplied = frozenset(ser_in.index)

        match supplied:
            case s if s == {'L', 'M', 'Q'}:
                self.P = self.L + self.M
                self.a = self.P + self.Q

            case s if s == {'L', 'M', 'a'}:
                self.P = self.L + self.M
                self.Q = self.a - self.P

            case s if s == {'L', 'M', 'coc'}:
                self.P = self.L + self.M
                self.Q = self.M / self.ROE
                self.a = self.P + self.Q

            case s if s == {'L', 'M', 'pq'}:
                self.P = self.L + self.M
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'L', 'P', 'Q'}:
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case s if s == {'L', 'P', 'a'}:
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case s if s == {'L', 'P', 'coc'}:
                self.M = self.P - self.L
                self.Q = self.M / self.ROE
                self.a = self.P + self.Q

            case s if s == {'L', 'P', 'pq'}:
                self.M = self.P - self.L
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'L', 'a', 'Q'}:
                self.P = self.a - self.Q
                self.M = self.P - self.L

            case s if s == {'L', 'Q', 'lr'}:
                self.P = self.L / self.LR
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case s if s == {'L', 'Q', 'coc'}:
                self.M = self.Q * self.ROE
                self.P = self.L + self.M
                self.a = self.P + self.Q

            case s if s == {'L', 'Q', 'pq'}:
                self.P = self.Q * self.PQ
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case s if s == {'L', 'a', 'lr'}:
                self.P = self.L / self.LR
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case s if s == {'L', 'a', 'coc'}:
                self.M = self.ROE / (1 + self.ROE) * (self.a - self.L)
                self.P = self.L + self.M
                self.Q = self.a - self.P

            case s if s == {'L', 'a', 'pq'}:
                self.P = self.a * self.PQ / (1 + self.PQ)
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case s if s == {'L', 'lr', 'coc'}:
                self.P = self.L / self.LR
                self.M = self.P - self.L
                self.Q = self.M / self.ROE
                self.a = self.P + self.Q

            case s if s == {'L', 'lr', 'pq'}:
                self.P = self.L / self.LR
                self.M = self.P - self.L
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'L', 'pq', 'coc'}:
                self.P = self.PQ / (self.PQ - self.ROE) * self.L
                self.M = self.P - self.L
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'P', 'M', 'Q'}:
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case s if s == {'P', 'M', 'a'}:
                self.L = self.P - self.M
                self.Q = self.a - self.P

            case s if s == {'P', 'M', 'coc'}:
                self.L = self.P - self.M
                self.Q = self.M / self.ROE
                self.a = self.P + self.Q

            case s if s == {'P', 'M', 'pq'}:
                self.L = self.P - self.M
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'M', 'a', 'Q'}:
                self.P = self.a - self.Q
                self.L = self.P - self.M

            case s if s == {'M', 'Q', 'lr'}:
                self.L = self.M * self.LR / (1 - self.LR)
                self.P = self.L + self.M
                self.a = self.P + self.Q

            case s if s == {'M', 'Q', 'pq'}:
                self.P = self.Q * self.PQ
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case s if s == {'M', 'a', 'lr'}:
                self.P = self.M / (1 - self.LR)
                self.L = self.P - self.M
                self.Q = self.a - self.P

            case s if s == {'M', 'a', 'coc'}:
                self.Q = self.M / self.ROE
                self.P = self.a - self.Q
                self.L = self.P - self.M

            case s if s == {'M', 'a', 'pq'}:
                self.P = self.a * self.PQ / (1 + self.PQ)
                self.L = self.P - self.M
                self.Q = self.a - self.P

            case s if s == {'M', 'lr', 'coc'}:
                self.P = self.M / (1 - self.LR)
                self.L = self.P - self.M
                self.Q = self.M / self.ROE
                self.a = self.P + self.Q

            case s if s == {'M', 'lr', 'pq'}:
                self.P = self.M / (1 - self.LR)
                self.L = self.P - self.M
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'M', 'pq', 'coc'}:
                self.P = self.PQ / self.ROE * self.M
                self.L = self.P - self.M
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'P', 'Q', 'lr'}:
                self.L = self.P * self.LR
                self.a = self.P + self.Q
                self.M = self.P - self.L

            case s if s == {'P', 'Q', 'coc'}:
                self.M = self.Q * self.ROE
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case s if s == {'P', 'a', 'lr'}:
                self.L = self.P * self.LR
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case s if s == {'P', 'a', 'coc'}:
                self.Q = self.a - self.P
                self.M = self.Q * self.ROE
                self.L = self.P - self.M

            case s if s == {'P', 'lr', 'coc'}:
                self.L = self.P * self.LR
                self.M = self.P - self.L
                self.Q = self.M / self.ROE
                self.a = self.P + self.Q

            case s if s == {'P', 'lr', 'pq'}:
                self.L = self.P * self.LR
                self.M = self.P - self.L
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q

            case s if s == {'P', 'pq', 'coc'}:
                self.Q = self.P / self.PQ
                self.a = self.P + self.Q
                self.M = self.Q * self.ROE
                self.L = self.P - self.M

            case s if s == {'a', 'Q', 'lr'}:
                self.P = self.a - self.Q
                self.L = self.P * self.LR
                self.M = self.P - self.L

            case s if s == {'a', 'Q', 'coc'}:
                self.P = self.a - self.Q
                self.M = self.Q * self.ROE
                self.L = self.P - self.M

            case s if s == {'Q', 'lr', 'coc'}:
                self.M = self.Q * self.ROE
                self.P = self.M / (1 - self.LR)
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case s if s == {'Q', 'lr', 'pq'}:
                self.P = self.Q * self.PQ
                self.L = self.P * self.LR
                self.M = self.P - self.L
                self.a = self.P + self.Q

            case s if s == {'Q', 'pq', 'coc'}:
                self.P = self.Q * self.PQ
                self.M = self.Q * self.ROE
                self.L = self.P - self.M
                self.a = self.P + self.Q

            case s if s == {'a', 'lr', 'coc'}:
                self.P = self.a * self.ROE / (self.ROE + 1 - self.LR)
                self.Q = self.a - self.P
                self.L = self.P * self.LR
                self.M = self.P - self.L

            case s if s == {'a', 'lr', 'pq'}:
                self.P = self.PQ * self.a / (1 + self.PQ)
                self.L = self.P * self.LR
                self.M = self.P - self.L
                self.Q = self.a - self.P

            case s if s == {'a', 'pq', 'coc'}:
                self.P = self.PQ * self.a / (1 + self.PQ)
                self.Q = self.a - self.P
                self.M = self.Q * self.ROE
                self.L = self.P - self.M

            case _:
                raise ValueError(f'Insoluble case: {tuple(sorted(supplied))}')

        # fill in / check the ratios
        self.ratios()

    @classmethod
    def test_cases(cls, L, P, a):
        """
        Run all test cases for a given set of loss, premium, and asset inputs.

        """
        M = P - L
        Q = a - P
        lr = L / P
        roe = M / Q
        pq = P / Q
        # canonical consistent pricing, keyed by the lowercase ratio spellings
        # used in the enumeration (lr/pq/coc) plus the amount names.
        consistent_pricing = pd.Series(
            [L, M, P, Q, a, lr, pq, roe],
            index=['L', 'M', 'P', 'Q', 'a', 'lr', 'pq', 'coc'],
        )

        # run through all options
        p = Pentagon()
        df = make_possible_pentagons()
        # add columns for each variable
        for c in cls.index:
            df[c] = None

        # iterate through all possible combinations. The enumeration names the
        # cost-of-capital ratio 'coc'; solve() spells that kwarg 'roe'.
        for i, r in df.iterrows():
            arg_dict = {('roe' if k == 'coc' else k): consistent_pricing[k]
                        for k in r.iloc[:3]}
            p.solve(**arg_dict)
            df.loc[i, cls.index] = p.values

        return df
