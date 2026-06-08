"""Tests for negative-support (signed) severity at the **Portfolio** level.

Covers ``dev/plan-negative-x-port.md`` -- the combine half of the P&L work:

- **Combine** independent signed-support units onto a shared signed grid via
  the FFT product (origin-at-0, ``ftagg_density`` is independent of each
  unit's ``x_min``) and a single F2 ``np.roll`` present step. The truncating
  ``ift`` of the non-negative path is replaced by a full-length ``irfft`` plus
  roll, so the wrapped negative tail is kept.
- **density_df** ``loss`` / ``p_total`` / ``p_{line}`` / ``F`` / ``S`` correct
  on signed support (and hence signed VaR/TVaR). Pricing columns (``add_exa``)
  are deferred -- a signed portfolio warns and falls back to F/S only.
- **Per-unit integrity**: units are driven on their own signed grids, so each
  unit object stays internally correct (moments, describe, plot).
- **Instrumentation**: two-sided ``_limits`` so ``plot`` shows the negative
  tail; ``describe`` finite for a mean-near-zero P&L.

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg``
(section PortPnL).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from aggregate import build
from aggregate.constants import DefectiveDistributionWarning


# Comparable-scale signed units: shared bs suits both, so the combine is exact.
# A: mean +250, B: mean -250 (each 50 claims, ssev 10*norm +/- 5, poisson).
# Sum: mean 0, var = 2 * 50 * (10^2 + 5^2) = 12500, sd ~ 111.803.
PNL_PROGRAM = '''port PnLpair
    agg A 50 claims ssev 10 * norm + 5 poisson
    agg B 50 claims ssev 10 * norm - 5 poisson'''

EXP_MEAN = 0.0
EXP_SD = (2 * 50 * (10 ** 2 + 5 ** 2)) ** 0.5  # 111.8034


@pytest.fixture(scope='module')
def pnl():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DefectiveDistributionWarning)
        return build(PNL_PROGRAM)


# ---------------------------------------------------------------------------
# Gate: signed detection.
# ---------------------------------------------------------------------------

def test_signed_gate(pnl):
    """A portfolio is signed iff some unit is; the gate drives every path."""
    assert pnl._signed() is True
    plain = build('''port Plain
        agg P 50 claims sev lognorm 100 cv 1 poisson
        agg Q 30 claims sev lognorm 80 cv 1 poisson''')
    assert plain._signed() is False


# ---------------------------------------------------------------------------
# Combine: moments add under independence.
# ---------------------------------------------------------------------------

def test_combine_moments(pnl):
    """Means add, variances add (independence); a symmetric pair has ~0 skew."""
    assert pnl.est_m == pytest.approx(EXP_MEAN, abs=1e-6)
    assert pnl.est_sd == pytest.approx(EXP_SD, rel=1e-4)
    assert pnl.est_skew == pytest.approx(0.0, abs=1e-4)


def test_mass_conserved(pnl):
    """``p_total`` sums to 1 -- the wrapped negative tail is NOT truncated."""
    assert float(pnl.density_df.p_total.sum()) == pytest.approx(1.0, abs=1e-9)


def test_signed_window_brackets_mass(pnl):
    """The realised window straddles 0 and brackets essentially all the mass."""
    lo, hi = pnl._signed_window
    assert lo < 0 < hi
    assert pnl.density_df.index[0] == pytest.approx(lo)
    # mass at the extreme buckets is negligible (window covers the support).
    edge = pnl.density_df.p_total.iloc[[0, -1]].abs().max()
    assert float(edge) < 1e-9


# ---------------------------------------------------------------------------
# Per-line marginals.
# ---------------------------------------------------------------------------

def test_per_line_marginals(pnl):
    """Each ``p_{line}`` integrates to 1 and carries the right (signed) mean."""
    df = pnl.density_df
    assert float(df.p_A.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(df.p_B.sum()) == pytest.approx(1.0, abs=1e-9)
    mean_A = float((df.p_A * df.loss).sum())
    mean_B = float((df.p_B * df.loss).sum())
    assert mean_A == pytest.approx(250.0, abs=0.5)
    assert mean_B == pytest.approx(-250.0, abs=0.5)
    # per-line means sum to the total mean.
    assert mean_A + mean_B == pytest.approx(float(pnl.est_m), abs=1e-3)


# ---------------------------------------------------------------------------
# Signed quantiles / risk measures read the loss index directly.
# ---------------------------------------------------------------------------

def test_signed_quantiles(pnl):
    """Low quantiles are negative, the median ~0, symmetric tails balance."""
    q_lo = float(pnl.q(0.001))
    q_md = float(pnl.q(0.5))
    q_hi = float(pnl.q(0.999))
    assert q_lo < 0 < q_hi
    assert q_md == pytest.approx(0.0, abs=pnl.bs)
    assert q_lo == pytest.approx(-q_hi, rel=1e-2)


def test_var_tvar_signed(pnl):
    """VaR/TVaR are well-defined on the signed grid (monotone index)."""
    assert float(pnl.var(0.99)) > 0
    # TVaR at a high level exceeds the VaR there.
    assert float(pnl.tvar(0.99)) >= float(pnl.var(0.99))


# ---------------------------------------------------------------------------
# Per-unit objects stay correct (driven on their own signed grids, plan 2c).
# ---------------------------------------------------------------------------

def test_per_unit_objects_correct(pnl):
    """Each unit is internally valid -- no false deficit, right signed mean."""
    a, b = pnl['A'], pnl['B']
    assert float(a.agg_density.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(b.agg_density.sum()) == pytest.approx(1.0, abs=1e-9)
    assert a.x_min < 0 and b.x_min < 0
    assert float(a.est_m) == pytest.approx(250.0, rel=1e-3)
    assert float(b.est_m) == pytest.approx(-250.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Instrumentation: two-sided plot range, plot smoke, describe finite.
# ---------------------------------------------------------------------------

def test_limits_two_sided(pnl):
    """``_limits('range')`` brackets the negative tail (the plot fix)."""
    lo, hi = pnl._limits('range')
    assert lo < 0 < hi
    assert lo == pytest.approx(-hi, rel=0.05)


def test_plot_smoke(pnl):
    """``plot`` renders without clipping the negative tail to a degenerate axis."""
    pnl.plot()
    import matplotlib.pyplot as plt
    plt.close('all')


def test_describe_finite(pnl):
    """``describe`` stays sane for a mean-zero P&L.

    No infinities, and the **EX error** column stays small everywhere because
    ``_noise_aware_rel_error`` degrades to absolute error when the theoretical
    mean is at noise level -- it does not explode via division by ~0. A signed
    portfolio reports **SD** (not CV), so the mean-exactly-zero total no longer
    shows a blown-up ``Est CV`` = sd/~0; the spread columns are finite.
    """
    d = pnl.describe
    arr = d.select_dtypes('number').to_numpy(dtype=float)
    assert not np.isinf(arr).any()
    err_ex = d['Err EX'].to_numpy(dtype=float)
    err_ex = err_ex[~np.isnan(err_ex)]
    assert np.all(np.abs(err_ex) < 1e-3)
    # signed portfolio -> SD spread columns, no CV columns
    assert any('SD' in str(c) for c in d.columns)
    assert not any('CV' in str(c) for c in d.columns)


def test_describe_mixed_signed_forces_sd_everywhere():
    """One signed unit forces the WHOLE describe table into SD.

    The spread choice (CV vs SD) is portfolio-wide: CV and SD cannot be mixed
    in one frame, so if any unit is signed the unsigned units render in SD too
    (via ``Aggregate._describe(force_sd=...)``) and the table aligns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DefectiveDistributionWarning)
        p = build('''port Mixed
            agg Signed dfreq [3] dsev [-2 -1 1 2]
            agg Plain  1 claim sev lognorm 10 cv .3 fixed''')
    d = p.describe
    # SD throughout, no CV anywhere
    assert any('SD' in str(c) for c in d.columns)
    assert not any('CV' in str(c) for c in d.columns)
    # every unit block AND total share one column layout (concat aligned)
    units = d.index.get_level_values('unit').unique().tolist()
    assert units == ['Signed', 'Plain', 'total']
    cols = list(d.columns)
    for u in units:
        assert list(d.xs(u, level='unit').columns) == cols
    # the unsigned unit's own SD is finite and matches its agg_sd
    plain = p['Plain']
    assert np.isclose(d.loc[('Plain', 'Agg'), 'SD'], plain.agg_sd)


def test_describe_unsigned_portfolio_uses_cv():
    """An all-unsigned portfolio keeps the CV layout unchanged."""
    p = build('''port Unsigned
        agg A 1 claim sev lognorm 10 cv .3 fixed
        agg B 1 claim sev lognorm 8 cv .2 fixed''')
    d = p.describe
    assert any('CV' in str(c) for c in d.columns)
    assert not any('SD' in str(c) for c in d.columns)


def test_info_reports_window(pnl):
    """``info`` advertises the realised signed window."""
    assert 'signed window' in pnl.info


# ---------------------------------------------------------------------------
# add_exa is deferred: warn + fall back to F/S only.
# ---------------------------------------------------------------------------

def test_add_exa_fallback():
    """``add_exa=True`` on a signed portfolio warns and writes F/S only."""
    p = build(PNL_PROGRAM, update=False)
    with pytest.warns(UserWarning, match='add_exa'):
        p.update(log2=16, bs=0, add_exa=True)
    assert 'F' in p.density_df and 'S' in p.density_df
    # the pricing columns are NOT present.
    assert 'exa_total' not in p.density_df


# ---------------------------------------------------------------------------
# Non-negative regression: the signed machinery never perturbs ordinary books.
# ---------------------------------------------------------------------------

def test_non_negative_regression():
    """A non-negative portfolio is unchanged by the signed code paths."""
    prog = '''port Reg
        agg A 50 claims sev lognorm 100 cv 2 poisson
        agg B 30 claims sev lognorm 80 cv 1.5 poisson'''
    p = build(prog)
    assert p._signed() is False
    assert p.density_df.index[0] == 0.0
    assert float(p.density_df.p_total.sum()) == pytest.approx(1.0, abs=1e-6)
    # no signed window attribute set on a non-signed book.
    assert getattr(p, '_signed_window', None) is None
    # _limits stays one-sided (positive support).
    lo, hi = p._limits('range')
    assert lo <= 0 < hi
