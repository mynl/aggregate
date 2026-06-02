"""Tests for the rationalized reinsurance reporting surface (1.0.0a19).

Covers ``dev/reins-reporting.md``:

- ``Aggregate.reins_density_df`` -- consistent columns regardless of which
  stages are configured; ``p_agg_subject`` present; mass conserved.
- ``Aggregate.reins_stats_df`` -- per-stage / per-view / per-basis (EX vs Est)
  moments; columns track the applied stages.
- ``Aggregate.reins_describe`` -- per-stage blocks; occurrence leads with
  ``Gross``, aggregate leads with ``Subject``; economic view (``EX``/``CV``/``Sk``
  hold the leading-view theoretic reference, ``Est`` is the per-view output,
  ``Change`` = impact / validation). The per-view EX-vs-Est rebucketing error
  (``linear`` preserves the directly-rebucketed mean; ``nearest`` to bs/2) is
  read from the internal ``_reins_view_stats`` frame.
- ``Portfolio.reins_{density,stats,describe}`` -- end-to-end gross/ceded/net;
  total means equal the sum of unit means per view; gross-only port -> None.

The DecL programs are mirrored in ``src/aggregate/agg/test_decl.agg``
(section Y).
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build


OCC_ONLY = (
    'agg RR.Occ 10 claims sev lognorm 100 cv 2 '
    'occurrence net of 0.8 so 250 xs 175 poisson'
)
AGG_ONLY = (
    'agg RR.Agg 10 claims sev lognorm 100 cv 2 poisson '
    'aggregate net of 500 xs 750'
)
BOTH = (
    'agg RR.Both 100 claims 5000 xs 0 sev lognorm 50 cv 1.5 '
    'occurrence net of 3500 po 4000 xs 1000 poisson '
    'aggregate net of 2000 xs 3000'
)

# Bounded severities for the EX-vs-Est numeric tests: a bounded aggregate has
# a negligible FFT grid deficit, so the EX (exact pre-bucket image moment) and
# Est (rebucketed) means differ only by the rebucketing scatter -- exactly what
# ``reins_bucket`` controls. The fractional-share off-grid cessions exercise
# the scatter. (Lognormal books carry a ~1e-9 deficit whose tail term swamps
# the rebucketing signal on the aggregate mean.)
OCC_BOUNDED = (
    'agg RR.OccB 10 claims sev 500 * beta 2 3 '
    'occurrence net of 0.8 so 150 xs 100 poisson'
)
AGG_BOUNDED = (
    'agg RR.AggB 10 claims sev 200 * beta 2 3 poisson '
    'aggregate net of 0.7 so 300 xs 400'
)

DENSITY_COLS = [
    'loss',
    'p_sev_gross', 'p_sev_ceded', 'p_sev_net',
    'p_agg_gross', 'p_agg_ceded_occ', 'p_agg_net_occ',
    'p_agg_subject', 'p_agg_ceded', 'p_agg_net',
]


# ----------------------------------------------------------------------------
# reins_density_df: consistent columns, mass conservation
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('prog', [OCC_ONLY, AGG_ONLY, BOTH])
def test_density_df_consistent_columns(prog):
    a = build(prog)
    rd = a.reins_density_df
    assert list(rd.columns) == DENSITY_COLS
    # the author's "third-to-last column" rename landed
    assert 'p_agg_subject' in rd.columns


@pytest.mark.parametrize('prog', [OCC_ONLY, AGG_ONLY, BOTH])
def test_density_df_mass_conserved(prog):
    a = build(prog)
    rd = a.reins_density_df
    for c in DENSITY_COLS[1:]:
        assert abs(float(rd[c].sum()) - 1.0) <= 1e-6, c


def test_no_reins_returns_none():
    a = build('agg RR.None 10 claims sev lognorm 100 cv 2 poisson')
    assert a.reins_density_df is None
    assert a.reins_stats_df is None
    assert a.reins_describe is None


# ----------------------------------------------------------------------------
# reins_stats_df: per-layer layering (Gross, layer.k, Ceded), empirical-only
# ----------------------------------------------------------------------------

def _stage_views(a):
    """view (occ/agg) -> ordered list of its layer columns (in column order).

    Column levels are ('view', 'layer'); 'view' holds occ/agg."""
    cols = a.reins_stats_df.columns
    out = {}
    for st, vw in zip(cols.get_level_values('view'),
                      cols.get_level_values('layer')):
        out.setdefault(st, [])
        if vw not in out[st]:
            out[st].append(vw)
    return out


def test_stats_df_columns_occ_only():
    # Gross, one column per occurrence layer, then Ceded + Net totals.
    sv = _stage_views(build(OCC_ONLY))
    assert list(sv) == ['occ']
    assert sv['occ'] == ['Gross', 'layer.1', 'Ceded', 'Net']


def test_stats_df_columns_agg_only():
    # Gross (the subject) lives under occ; agg block has no Subject column.
    sv = _stage_views(build(AGG_ONLY))
    assert list(sv) == ['occ', 'agg']
    assert sv['occ'] == ['Gross']
    assert sv['agg'] == ['layer.1', 'Ceded', 'Net']


def test_stats_df_columns_both():
    # occ block before agg block; no agg Subject column
    sv = _stage_views(build(BOTH))
    assert list(sv) == ['occ', 'agg']
    assert sv['occ'] == ['Gross', 'layer.1', 'Ceded', 'Net']
    assert sv['agg'] == ['layer.1', 'Ceded', 'Net']


def test_stats_df_column_level_names():
    """Column levels are (view, layer); view holds occ/agg (no EX/Est basis)."""
    assert build(BOTH).reins_stats_df.columns.names == ['view', 'layer']


def test_stats_df_meta_rows():
    """Each layer carries share / limit / attach / pr_attach; non-layer
    columns NaN on those rows."""
    a = build('agg A 2 claims sev 128 * uniform '
              'occurrence net of 50% so 32 xs 32 poisson')
    rs = a.reins_stats_df
    assert rs.loc[('meta', 'share'), ('occ', 'layer.1')] == pytest.approx(0.5)
    assert rs.loc[('meta', 'limit'), ('occ', 'layer.1')] == pytest.approx(32)
    assert rs.loc[('meta', 'attach'), ('occ', 'layer.1')] == pytest.approx(32)
    # P(X > 32) for uniform[0, 128]
    assert rs.loc[('meta', 'pr_attach'), ('occ', 'layer.1')] == pytest.approx(0.75)
    # Gross carries the claim-count-weighted policy terms (share 1)
    assert rs.loc[('meta', 'share'), ('occ', 'Gross')] == pytest.approx(1.0)


def test_stats_df_pricing_meta_rows():
    """pr_detach, pr_loss and lol for an occurrence layer + gross book."""
    a = build('agg A 2 claims sev 128 * uniform '
              'occurrence net of 32 xs 32 and 64 xs 64 poisson', bs=1/32, log2=16)
    rs = a.reins_stats_df
    # pr_detach = P(X >= attach + limit): layer.1 (32 xs 32) -> P(X > 64) = 0.5
    assert rs.loc[('meta', 'pr_detach'), ('occ', 'layer.1')] == pytest.approx(0.5, abs=1e-3)
    # unlimited gross severity -> no detachment
    assert np.isnan(rs.loc[('meta', 'pr_detach'), ('occ', 'Gross')])
    # pr_loss = P(aggregate > 0); Poisson(2) gross -> 1 - e**-2
    assert rs.loc[('meta', 'pr_loss'), ('occ', 'Gross')] == pytest.approx(
        1 - np.exp(-2), abs=1e-3)
    # lol = layer agg mean / placed limit; layer.1 = 40 / 32
    lol = rs.loc[('meta', 'lol'), ('occ', 'layer.1')]
    aggm = rs.loc[('agg', 'mean'), ('occ', 'layer.1')]
    assert lol == pytest.approx(aggm / 32, rel=1e-6)
    # Net total has no limit -> lol NaN
    assert np.isnan(rs.loc[('meta', 'lol'), ('occ', 'Net')])


def test_stats_df_ex1_equals_mean():
    """ex1 duplicates mean for easy regex filtering."""
    rs = build(OCC_ONLY).reins_stats_df
    for comp in ('freq', 'sev', 'agg'):
        e = rs.loc[(comp, 'ex1'), ('occ', 'Gross')]
        m = rs.loc[(comp, 'mean'), ('occ', 'Gross')]
        assert e == pytest.approx(m)


def test_stats_df_output_row():
    """The output 0/1 row flags each stage's output view: occ + agg => two 1s;
    Gross carries the 1 when there is no occurrence program."""
    both = build('agg A 2 claims sev 128 * uniform '
                 'occurrence net of 96 xs 32 poisson aggregate ceded to 15 xs 40')
    out = both.reins_stats_df.loc[('meta', 'output')]
    assert out.sum() == pytest.approx(2.0)
    assert out[('occ', 'Net')] == 1.0          # occ output (net of)
    assert out[('agg', 'Ceded')] == 1.0        # agg output (ceded to)
    agg_only = build('agg A 2 claims sev 128 * uniform poisson '
                     'aggregate net of 50 xs 60')
    out2 = agg_only.reins_stats_df.loc[('meta', 'output')]
    assert out2[('occ', 'Gross')] == 1.0       # no occ => Gross is the subject
    assert out2[('agg', 'Net')] == 1.0


def test_stats_df_freq_gross_full_moments():
    """Gross frequency carries all moments, matching stats_df['mixed']
    (issue 3): not just the mean."""
    a = build(OCC_ONLY)
    rs = a.reins_stats_df
    mixed = a.stats_df['mixed']
    for measure in ('ex1', 'ex2', 'ex3', 'mean', 'cv'):
        got = rs.loc[('freq', measure), ('occ', 'Gross')]
        exp = mixed[('freq', measure)]
        assert got == pytest.approx(exp), measure


def test_stats_df_occ_layer_conditional_severity():
    """Each occurrence layer's severity is conditional: unconditional layer
    severity divided by P(attach), with freq the penetrating count."""
    a = build('agg A 2 claims sev 128 * uniform '
              'occurrence net of 96 xs 32 poisson', bs=1/32, log2=16)
    rs = a.reins_stats_df
    pr = rs.loc[('meta', 'pr_attach'), ('occ', 'layer.1')]
    # conditional sev = unconditional total ceded sev / pr (single layer)
    uncond = rs.loc[('sev', 'mean'), ('occ', 'Ceded')]
    cond = rs.loc[('sev', 'mean'), ('occ', 'layer.1')]
    assert cond == pytest.approx(uncond / pr, rel=1e-6)
    # conditional count n' = n * pr
    assert rs.loc[('freq', 'mean'), ('occ', 'layer.1')] == pytest.approx(2 * pr)


def test_stats_df_ceded_plus_net_equals_gross_sev():
    """Unconditional occ Ceded sev + Net sev == Gross sev."""
    a = build('agg A 2 claims sev 128 * uniform '
              'occurrence net of 32 xs 32 and 64 xs 64 poisson')
    rs = a.reins_stats_df
    g = rs.loc[('sev', 'mean'), ('occ', 'Gross')]
    c = rs.loc[('sev', 'mean'), ('occ', 'Ceded')]
    nt = rs.loc[('sev', 'mean'), ('occ', 'Net')]
    assert c + nt == pytest.approx(g, rel=1e-6)


def test_stats_df_agg_layer_means_add_to_ceded_total():
    """Aggregate layer means sum to the Ceded aggregate total."""
    a = build('agg A 2 claims sev 128 * uniform '
              'occurrence net of 32 xs 32 and 64 xs 64 poisson')
    rs = a.reins_stats_df
    layer_cols = [c for c in rs.columns if c[0] == 'occ' and 'layer' in c[1]]
    layer_sum = sum(rs.loc[('agg', 'mean'), c] for c in layer_cols)
    total = rs.loc[('agg', 'mean'), ('occ', 'Ceded')]
    assert layer_sum == pytest.approx(total, rel=1e-6)


# ----------------------------------------------------------------------------
# reins_describe: per-stage layout; gross/subject lead
# ----------------------------------------------------------------------------

DESCRIBE_COLS = ['EX', 'Est EX', 'Change EX',
                 'CV', 'Est CV', 'Change CV', 'Sk', 'Est Sk']


def test_describe_occ_leads_gross():
    d = build(OCC_ONLY).reins_describe
    occ = d.xs('occ', level='stage')
    # view / component index labels are lower-case (match the other frames)
    assert occ.index.get_level_values('view')[0] == 'gross'
    # occurrence block carries freq / sev / agg components
    assert set(occ.index.get_level_values('component')) == {'freq', 'sev', 'agg'}
    # same eight columns as Aggregate.describe
    assert list(d.columns) == DESCRIBE_COLS


def test_describe_agg_leads_subject():
    d = build(AGG_ONLY).reins_describe
    agg = d.xs('agg', level='stage')
    assert agg.index.get_level_values('view')[0] == 'subject'
    # aggregate block is agg only (sev N/A, freq degenerate)
    assert set(agg.index.get_level_values('component')) == {'agg'}


def test_describe_both_has_both_stages():
    d = build(BOTH).reins_describe
    assert set(d.index.get_level_values('stage')) == {'occ', 'agg'}


def test_describe_reference_constant_down_component():
    """``EX`` / ``CV`` / ``Sk`` hold the leading-view (Gross / Subject) theoretic
    reference, so they are identical across the views of a given stage+component
    (the economic view of ``describe``)."""
    d = build(BOTH).reins_describe
    for stage in ('occ', 'agg'):
        blk = d.xs(stage, level='stage')
        for comp in blk.index.get_level_values('component').unique():
            sub = blk.xs(comp, level='component')
            for col in ('EX', 'CV', 'Sk'):
                vals = sub[col].to_numpy(dtype=float)
                ref = vals[0]
                # all rows share the leading-view reference (NaN-safe compare)
                assert np.allclose(vals, ref, rtol=1e-12, atol=1e-12,
                                   equal_nan=True), (stage, comp, col)


def test_describe_change_is_impact_vs_gross():
    """On a ceded/net row, ``Change EX = (Est EX - EX) / EX`` measures the
    cession's impact relative to the gross/subject reference; the leading-view
    row's Change is the validation/rebucketing error (~0 under linear)."""
    a = build(OCC_BOUNDED, reins_bucket='linear')
    occ = a.reins_describe.xs('occ', level='stage')
    # gross/agg row: Est is the model gross, reference is theoretic gross ->
    # Change is the validation error, ~0 for a bounded book under linear.
    g = occ.loc[('gross', 'agg')]
    assert abs(float(g['Change EX'])) <= 1e-6
    # net/agg row: reference is still gross, Est is the net output, so Change
    # reproduces (Est - EX) / EX and is materially negative (cession reduces).
    n = occ.loc[('net', 'agg')]
    expected = (float(n['Est EX']) - float(n['EX'])) / float(n['EX'])
    assert abs(float(n['Change EX']) - expected) <= 1e-9
    assert float(n['Change EX']) < 0.0


def test_describe_freq_unconditional_and_gross_nan():
    """Frequency on the model-output (Est) basis: the gross row is NaN
    (mirrors describe), ceded / net carry the unconditional mean E[N] only
    (so freq * sev == agg per view), and cv / skew stay NaN."""
    a = build(OCC_BOUNDED, reins_bucket='linear')
    occ = a.reins_describe.xs('occ', level='stage')
    # gross freq Est entirely NaN
    g = occ.loc[('gross', 'freq')]
    assert np.isnan(float(g['Est EX']))
    assert np.isnan(float(g['Est CV']))
    assert np.isnan(float(g['Est Sk']))
    # ceded / net freq Est mean is the unconditional E[N] (= gross freq mean),
    # so the freq change is ~0; cv / skew Est stay NaN.
    en = float(a.n)
    for view in ('ceded', 'net'):
        f = occ.loc[(view, 'freq')]
        assert abs(float(f['Est EX']) - en) <= 1e-9, view
        assert abs(float(f['Change EX'])) <= 1e-9, view
        assert np.isnan(float(f['Est CV'])), view
        assert np.isnan(float(f['Est Sk'])), view
    # freq * sev == agg on the Est basis for the ceded view (means multiply)
    cf = float(occ.loc[('ceded', 'freq'), 'Est EX'])
    cs = float(occ.loc[('ceded', 'sev'), 'Est EX'])
    ca = float(occ.loc[('ceded', 'agg'), 'Est EX'])
    assert abs(cf * cs - ca) <= 1e-6 * max(1.0, abs(ca))


# ----------------------------------------------------------------------------
# EX vs Est rebucketing error, surfaced via the public reins_describe columns:
# linear preserves the directly-rebucketed mean; nearest is within bs/2.
# ----------------------------------------------------------------------------

def _ex_est(a, stage, view, comp):
    """Per-view exact (EX) vs rebucketed (Est) mean from the internal
    ``_reins_view_stats`` frame -- the rebucketing-error source. (``reins_describe``
    itself now holds the *gross/subject* reference in its ``EX`` column, so its
    EX-vs-Est is an economic change, not a per-view bucketing error.)"""
    rs = a._reins_view_stats
    ex = float(rs.loc[(comp.lower(), 'mean'), (stage, view.lower(), 'EX')])
    est = float(rs.loc[(comp.lower(), 'mean'), (stage, view.lower(), 'Est')])
    return ex, est


def test_linear_preserves_occ_severity_mean():
    """Linear mass-split preserves the per-claim (severity) first moment of
    every occurrence view exactly (no rebucketing bias)."""
    a = build(OCC_BOUNDED, reins_bucket='linear')
    for view in ('Gross', 'Ceded', 'Net'):
        ex, est = _ex_est(a, 'occ', view, 'Sev')
        assert abs(est - ex) <= 1e-8 * max(1.0, abs(ex)), view


def test_linear_preserves_agg_stage_mean():
    """The aggregate cover rebuckets the aggregate directly; linear preserves
    its mean exactly for subject, ceded and net."""
    a = build(AGG_BOUNDED, reins_bucket='linear')
    for view in ('Subject', 'Ceded', 'Net'):
        ex, est = _ex_est(a, 'agg', view, 'Agg')
        assert abs(est - ex) <= 1e-8 * max(1.0, abs(ex)), view


def test_nearest_within_half_bucket_occ_severity():
    a = build(OCC_BOUNDED, reins_bucket='nearest')
    for view in ('Gross', 'Ceded', 'Net'):
        ex, est = _ex_est(a, 'occ', view, 'Sev')
        assert abs(est - ex) <= a.bs / 2 + 1e-9, view


def test_nearest_within_half_bucket_agg_stage():
    a = build(AGG_BOUNDED, reins_bucket='nearest')
    for view in ('Subject', 'Ceded', 'Net'):
        ex, est = _ex_est(a, 'agg', view, 'Agg')
        assert abs(est - ex) <= a.bs / 2 + 1e-9, view


# ----------------------------------------------------------------------------
# Portfolio: end-to-end gcn
# ----------------------------------------------------------------------------

PORT_RE = (
    'port RR.Port '
    'agg A 100 claims 5000 xs 0 sev lognorm 50 cv 1.5 '
    'occurrence net of 2000 xs 1000 poisson '
    'agg B 50 claims sev lognorm 100 cv 2 poisson aggregate net of 1000 xs 2000 '
    'agg C 20 claims sev lognorm 30 cv 1 poisson'
)
PORT_GROSS = (
    'port RR.Gross '
    'agg A 10 claims sev lognorm 100 cv 2 poisson '
    'agg B 5 claims sev lognorm 50 cv 1 poisson'
)


def test_port_density_df_columns_and_mass():
    p = build(PORT_RE)
    rd = p.reins_density_df
    assert list(rd.columns) == ['loss', 'p_agg_gross', 'p_agg_ceded', 'p_agg_net']
    for c in ['p_agg_gross', 'p_agg_ceded', 'p_agg_net']:
        assert abs(float(rd[c].sum()) - 1.0) <= 1e-6, c


def test_port_total_means_sum_of_units():
    """Means add under convolution: portfolio total == sum of unit end-to-end
    means, per view."""
    p = build(PORT_RE)
    rs = p.reins_stats_df
    for v in ('gross', 'ceded', 'net'):
        tot = float(rs.loc[(v, 'mean'), 'total'])
        usum = float(sum(rs.loc[(v, 'mean'), a.name] for a in p))
        assert abs(tot - usum) <= 1e-4 * max(1.0, abs(tot)), v


def test_port_describe_alignment():
    p = build(PORT_RE)
    d = p.reins_describe
    units = set(d.index.get_level_values('unit'))
    # ceding units A, B and the total block are present; non-ceding C is not
    assert 'total' in units
    assert {'A', 'B'} <= units
    assert 'C' not in units
    assert list(d.columns) == DESCRIBE_COLS


def test_port_gross_only_returns_none():
    g = build(PORT_GROSS)
    assert g.reins_density_df is None
    assert g.reins_stats_df is None
    assert g.reins_describe is None
