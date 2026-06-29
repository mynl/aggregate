r"""The insurance View over the domain-free leg kernel.

The leg kernel (:mod:`aggregate.legs`) knows only ``(X, Y) -> legs ->
distributions``; it carries no insurance vocabulary. **All** of the insurance
labelling lives here, as a thin *composition* over a
:class:`~aggregate.legs.LegSet` -- never a mixin on :class:`~aggregate.legs.Leg`
(a per-leg mixin would re-attach the domain semantics the kernel deliberately
sheds). A View

* **owns the labels** -- the ``name -> (perspective, category)`` map and the
  ``category -> kind`` rollup (``premium`` -> ``consideration``;
  ``loss`` / ``expense`` -> ``obligation``; ``underwriting`` -> ``margin``); and
* **wraps a leg set and a source** as an :class:`InsuranceView`, caching the one
  kernel evaluation (``distributions`` + ``stats_df``) that every exhibit reads.

Reinsurance, variable rating and reinstatement are all Views differing only in
which leg(s) a feature overrides; they assemble a leg set, hand it to an
:class:`InsuranceView`, and build their gross / ceded / net exhibits off the
cached evaluation. The grouping helpers (``gcn_assemble_column``, the fixed
summary table) stay in :mod:`aggregate._pnl` / the analysis classes for now;
this module is the shared vocabulary they agree on.

Submodule access only (no top-level re-export)::

    from aggregate._insurance_view import InsuranceView, perspective_of, kind_of
"""

from __future__ import annotations

import numpy as np

__all__ = ['InsuranceView', 'PERSPECTIVES', 'CATEGORIES', 'KIND_OF',
           'perspective_of', 'category_of', 'kind_of', 'label_of']


#: The reinsurance waterfall perspectives (a leg books on exactly one).
PERSPECTIVES = ('gross', 'ceded', 'net', 'ceded_agg', 'net_agg', 'total_ceded')

#: The accounting categories a leg falls into.
CATEGORIES = ('premium', 'loss', 'expense', 'underwriting')

#: The fixed P&L rollup of a category to its signed-summary **kind** (the
#: ``Consideration + Obligation = Margin`` layout). ``premium`` is money in at
#: inception (consideration); ``loss`` / ``expense`` are the obligations borne;
#: ``underwriting`` is the resulting margin.
KIND_OF = {
    'premium': 'consideration',
    'loss': 'obligation',
    'expense': 'obligation',
    'underwriting': 'margin',
}

#: Irregular leg names whose perspective / category the suffix rules below would
#: misread. ``commission`` and the reinstatement premium are cessions; the
#: ``unlimited_ceded_loss`` axis is the (informational) gross recovery.
_IRREGULAR = {
    'commission': ('ceded', 'expense'),
    'reinstatement_premium': ('ceded', 'premium'),
    'unlimited_ceded_loss': ('ceded', 'loss'),
}


def perspective_of(name: str) -> str:
    """The waterfall perspective a leg books on (``gross`` / ``ceded`` / ...).

    Inferred from the leg name prefix; the multi-word perspectives
    (``total_ceded``, ``ceded_agg``, ``net_agg``) are tested before the bare
    ``ceded`` / ``net`` so the longer match wins.
    """
    if name in _IRREGULAR:
        return _IRREGULAR[name][0]
    if name.startswith('gross'):
        return 'gross'
    if name.startswith('total_ceded'):
        return 'total_ceded'
    if name.startswith('ceded_agg'):
        return 'ceded_agg'
    if name.startswith('net_agg'):
        return 'net_agg'
    if name.startswith('ceded'):
        return 'ceded'
    if name.startswith('net'):
        return 'net'
    raise ValueError(f'cannot infer a perspective for leg {name!r}.')


def category_of(name: str) -> str:
    """The accounting category of a leg (``premium`` / ``loss`` / ``expense`` /
    ``underwriting``), inferred from the leg-name suffix."""
    if name in _IRREGULAR:
        return _IRREGULAR[name][1]
    if name.endswith('_uw') or name.endswith('underwriting'):
        return 'underwriting'
    if name.endswith('_premium'):
        return 'premium'
    if name.endswith('_expense') or name == 'commission':
        return 'expense'
    if name.endswith('_loss'):
        return 'loss'
    raise ValueError(f'cannot infer a category for leg {name!r}.')


def kind_of(name: str) -> str:
    """The signed-summary kind of a leg (``consideration`` / ``obligation`` /
    ``margin``) -- its category rolled up through :data:`KIND_OF`."""
    return KIND_OF[category_of(name)]


def label_of(name: str) -> tuple:
    """``(perspective, category)`` for a leg name (the full insurance label)."""
    return perspective_of(name), category_of(name)


class InsuranceView:
    """Composition of a :class:`~aggregate.legs.LegSet` with a source and labels.

    The shared backbone of the insurance analyses: it owns the one kernel
    evaluation -- pushing every leg over the source (:attr:`distributions`) and
    the exact per-leg moments (:attr:`stats_df`) -- and the
    name -> ``(perspective, category, kind)`` vocabulary the exhibits read. The
    analysis classes hold one of these and build their gross / ceded / net
    tables off it; "1-D vs 2-D" is just which ``source`` was handed in (a
    :class:`~aggregate.legs.GraphSource` or a full
    :class:`~aggregate.bivariate.BivariateDistribution`).

    Parameters
    ----------
    leg_set : LegSet
        The assembled accounting legs (one feature may have overridden a map).
    source : object
        Anything satisfying the kernel source protocol
        (``pushforward`` / ``transformed_moments``).
    point_masses : dict, optional
        ``{name: float}`` legs that are **deterministic constants** (e.g. a fixed
        gross premium). These are exact one-point distributions, kept off the
        source so they stay byte-exact rather than rebucketed; they join
        :attr:`distributions` and :attr:`stats_df` after the kernel pass.
    """

    def __init__(self, leg_set, source, *, point_masses=None):
        self.leg_set = leg_set
        self.source = source
        self.point_masses = dict(point_masses) if point_masses else {}
        self._distributions = None
        self._stats = None

    # -- the one kernel evaluation, cached ------------------------------
    @property
    def distributions(self) -> dict:
        """``{name: GridDistribution}`` for every leg (kernel pushforward)."""
        if self._distributions is None:
            from ._grid_distribution import GridDistribution
            d = self.leg_set.distributions(self.source)
            for name, value in self.point_masses.items():
                d[name] = GridDistribution(
                    np.array([float(value)]), np.array([1.0]),
                    name=name, is_loss_value=True)
            self._distributions = d
        return self._distributions

    @property
    def stats_df(self):
        """Exact per-leg moments (the "EX" column); a point mass is certain."""
        if self._stats is None:
            df = self.leg_set.stats_df(self.source)
            for name, value in self.point_masses.items():
                df.loc[name] = {'mass': 1.0, 'mean': float(value), 'var': 0.0,
                                'sd': 0.0, 'cv': 0.0, 'skew': np.nan}
            self._stats = df
        return self._stats

    def exact(self, name) -> tuple:
        """``(mean, sd, skew)`` of a leg from the exact :attr:`stats_df`."""
        import pandas as pd
        row = self.stats_df.loc[name]
        sk = row.get('skew', np.nan)
        return (float(row['mean']), float(row['sd']),
                float(sk) if pd.notna(sk) else 0.0)

    # -- the vocabulary -------------------------------------------------
    @staticmethod
    def perspective(name):
        """The waterfall perspective of a leg (see :func:`perspective_of`)."""
        return perspective_of(name)

    @staticmethod
    def category(name):
        """The accounting category of a leg (see :func:`category_of`)."""
        return category_of(name)

    @staticmethod
    def kind(name):
        """The signed-summary kind of a leg (see :func:`kind_of`)."""
        return kind_of(name)
