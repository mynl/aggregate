"""Tests for the massive (disk-backed) bivariate kernel (``dev/plan-bv.md`` §8).

Phase 1 (§8.1--.2): the out-of-core 3-pass kernel
:func:`aggregate._aggregate_compute_massive.massive_bivariate_convolution`
reproduces the in-core ``update_work`` density to ``1e-12`` across all three
modes (copula, discrete, netceded), signed axes (``i0 != 0``), shifted windows
(``j0 != 0``), and chunk shapes including non-divisors; the pass-3
accumulators (marginals, total mass, mixed raw moments) match the in-core
reductions.
"""

import numpy as np
import pytest
import scipy.sparse as ssp

zarr = pytest.importorskip('zarr')

from aggregate import build
from aggregate._aggregate_compute_massive import massive_bivariate_convolution

# Bleeding-edge disk-backed bivariate kernel and the heaviest cases in the
# suite; quarantined from the fast local loop (`-m 'not slow'`), still run in
# full/CI.
pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------
# reference programs (mirroring tests/test_bivariate.py)
# ----------------------------------------------------------------------

COPULA_PROG = '''bivariate MV 25 claims
    agg A dfreq [0 1] [.3 .7] sev lognorm 40 cv 1.2
    agg B dfreq [0 1] [.5 .5] sev lognorm 60 cv 1.5
    copula gumbel 0.4
    poisson'''

# signed ``ssev`` axes: i0 != 0 (negative severity lay-in) and j0 < 0
# (two-sided centred windows).
SIGNED_PROG = '''bivariate SIGNED
    200 claims
    agg A dfreq[1] ssev uniform - .3
    agg B dfreq[1] ssev uniform - .5
    mixed gamma .5'''

# non-negative book far from 0: x_min > 0 so j0 > 0 with i0 == 0.
FAR_PROG = '''bivariate FAR 500 claims
    agg AL 1 claim  sev 10 * uniform fixed
    agg GL 1 claim  sev 20 * uniform fixed
    poisson'''

DISCRETE_PROG = ('bv DBV dfreq [1 2 3] [.5 .3 .2] '
                 'dbvsev [0 1 2] [0 5 10] [[.4 .1 .0] [.1 .1 .1] [.0 .05 .15]]')

NC_PROG = ('agg NC 8 claims sev 300 * beta 2 3 '
           'occurrence net of 0.7 po 60 xs 40 poisson')


def _copula_kernel_kwargs(mv, store_dir, **overrides):
    """Kernel arguments read off an in-core-updated copula/discrete bv."""
    kw = dict(S=mv._S, freq_pgf=mv.frequency.freq_pgf, en=mv.en,
              N0=mv._nout[0], N1=mv._nout[1],
              bs0=mv.bs[0], bs1=mv.bs[1],
              i0=tuple(mv._i0), j0=tuple(mv._j0), mlog2=tuple(mv._mlog2),
              xs0=mv.axis_xs[0], xs1=mv.axis_xs[1],
              store_dir=str(store_dir))
    kw.update(overrides)
    return kw


def _check_result(res, mv, atol=1e-12):
    """Density, marginals, mass and mixed moments match the in-core update."""
    dens = res.density[:]
    assert dens.shape == mv.density.shape
    assert np.allclose(dens, mv.density, atol=atol)
    m0, m1 = mv.marginals
    assert np.allclose(res.marg0, m0, atol=atol)
    assert np.allclose(res.marg1, m1, atol=atol)
    assert res.total_mass == pytest.approx(float(mv.density.sum()), abs=1e-12)
    assert np.allclose(res.raw_moments, mv.moments(3).to_numpy(), rtol=1e-9)


@pytest.mark.parametrize('chunks', [(64, 64), (37, 100), (512, 512)])
def test_kernel_matches_incore_copula(tmp_path, chunks):
    """§8.1: bit-level match on the copula mode, chunk sweep incl. non-divisors."""
    mv = build(COPULA_PROG)
    mv.update(log2=16)
    res = massive_bivariate_convolution(
        **_copula_kernel_kwargs(mv, tmp_path / 'bv',
                                row_chunk=chunks[0], col_chunk=chunks[1]))
    _check_result(res, mv)


def test_kernel_matches_incore_signed(tmp_path):
    """§8.2: signed axes -- i0 != 0 lay-in wrap and negative j0 window roll."""
    mv = build(SIGNED_PROG)
    mv.update(log2=16)
    assert any(i > 0 for i in mv._i0), 'test premise: signed axes expected'
    assert any(j < 0 for j in mv._j0), 'test premise: negative window origin'
    res = massive_bivariate_convolution(
        **_copula_kernel_kwargs(mv, tmp_path / 'bv'))
    _check_result(res, mv)


def test_kernel_matches_incore_far_from_zero(tmp_path):
    """§8.2: non-negative mass far from 0 -- positive j0 with i0 == 0."""
    mv = build(FAR_PROG)
    mv.update(log2=16)
    assert all(i == 0 for i in mv._i0)
    assert all(j > 0 for j in mv._j0), 'test premise: shifted window expected'
    res = massive_bivariate_convolution(
        **_copula_kernel_kwargs(mv, tmp_path / 'bv'))
    _check_result(res, mv)


def test_kernel_matches_incore_discrete(tmp_path):
    """§8.1: the given-matrix (dbvsev) mode."""
    mv = build(DISCRETE_PROG)
    res = massive_bivariate_convolution(
        **_copula_kernel_kwargs(mv, tmp_path / 'bv'))
    _check_result(res, mv)


def test_kernel_matches_incore_netceded(tmp_path):
    """§8.1: the netceded comonotone mode (0-based grids, inner-agg pgf)."""
    mv = build(f'netceded {NC_PROG}')
    a = mv._nc_agg
    n0, n1 = mv.density.shape
    pad = a.padding
    mlog2 = (int(np.log2(n0)) + pad, int(np.log2(n1)) + pad)
    res = massive_bivariate_convolution(
        S=mv._S, freq_pgf=a.frequency.freq_pgf, en=a.n,
        N0=n0, N1=n1, bs0=mv.bs[0], bs1=mv.bs[1],
        i0=(0, 0), j0=(0, 0), mlog2=mlog2,
        xs0=mv.axis_xs[0], xs1=mv.axis_xs[1],
        store_dir=str(tmp_path / 'bv'))
    dens = res.density[:]
    assert np.allclose(dens, mv.density, atol=1e-12)
    m0, m1 = mv.marginals
    assert np.allclose(res.marg0, m0, atol=1e-12)
    assert np.allclose(res.marg1, m1, atol=1e-12)


def test_kernel_sparse_severity_matches_dense(tmp_path):
    """A scipy.sparse S (the netceded representation at scale) is identical."""
    mv = build(f'netceded {NC_PROG}')
    a = mv._nc_agg
    n0, n1 = mv.density.shape
    mlog2 = (int(np.log2(n0)) + a.padding, int(np.log2(n1)) + a.padding)
    kw = dict(freq_pgf=a.frequency.freq_pgf, en=a.n,
              N0=n0, N1=n1, bs0=mv.bs[0], bs1=mv.bs[1],
              i0=(0, 0), j0=(0, 0), mlog2=mlog2,
              xs0=mv.axis_xs[0], xs1=mv.axis_xs[1])
    dense = massive_bivariate_convolution(
        S=mv._S, store_dir=str(tmp_path / 'dense'), **kw)
    sparse = massive_bivariate_convolution(
        S=ssp.csr_matrix(mv._S), store_dir=str(tmp_path / 'sparse'), **kw)
    assert np.array_equal(dense.density[:], sparse.density[:])
    assert dense.total_mass == sparse.total_mass


def test_kernel_en_zero_point_mass(tmp_path):
    """en == 0 short-circuits to a point mass at physical (0, 0)."""
    N = 16
    # signed windows containing physical 0 (j0 < 0 <=> x_min < 0)
    j0 = (-3, -2)
    res = massive_bivariate_convolution(
        S=np.ones((2, 2)) / 4, freq_pgf=None, en=0,
        N0=N, N1=N, bs0=1.0, bs1=1.0, i0=(0, 0), j0=j0, mlog2=(5, 5),
        store_dir=str(tmp_path / 'bv'))
    dens = res.density[:]
    r, c = (-j0[0]) % N, (-j0[1]) % N
    assert dens[r, c] == 1.0
    assert dens.sum() == 1.0
    assert res.total_mass == 1.0
    # the point sits at physical 0 on both axes: first moments vanish
    assert res.raw_moments[1, 0] == pytest.approx(res.xs0[r])
    assert res.xs0[r] == pytest.approx(0.0)
    assert res.xs1[c] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Phase 2 (§8.3, §8.5, §8.6): update(store_dir=), the container, reopen
# ----------------------------------------------------------------------

def test_update_store_dir_matches_incore_copula(tmp_path):
    """update(store_dir=...) reproduces the in-core update end to end.

    padding=1 is forced so the FFT buffers match the in-core run exactly
    (the massive default is padding=0, tested separately below).
    """
    from aggregate.bivariate import MassiveBivariateDistribution
    mv_ref = build(COPULA_PROG)
    mv_ref.update(log2=16)
    mv = build(COPULA_PROG)
    mv.update(log2=16, padding=1, store_dir=str(tmp_path / 'bv'))
    assert mv.density is None
    bd = mv.bivariate
    assert isinstance(bd, MassiveBivariateDistribution)
    # accumulator surface vs in-core
    m0, m1 = mv.marginals
    r0, r1 = mv_ref.marginals
    assert np.allclose(m0, r0, atol=1e-12)
    assert np.allclose(m1, r1, atol=1e-12)
    assert np.allclose(mv.moments(3).to_numpy(),
                       mv_ref.moments(3).to_numpy(), rtol=1e-9)
    assert mv.corr == pytest.approx(mv_ref.corr, abs=1e-9)
    assert mv.deficit == pytest.approx(mv_ref.deficit, abs=1e-10)
    # the lazy density view slices like the in-core array
    assert np.allclose(bd.density[100:110, :], mv_ref.density[100:110, :],
                       atol=1e-12)
    # reporting surface runs
    assert 'disk-backed' in repr(mv)
    assert mv.info
    assert mv.summary_df.shape[0] == 7
    assert np.isfinite(mv.dependency_df.loc['Sev', 'corr'])


def test_update_store_dir_padding0_default_clean(tmp_path):
    """The massive default padding=0 stays clean and matches in-core padding=0.

    The measured window already covers the support to 10**-window_nines, so
    dropping the padding costs ~nothing (plan-bv §4.4); the marginal mean at
    the matched grid is identical to the in-core run (the residual difference
    vs theory is bs-discretisation, common to both paths).
    """
    mv_ref = build(COPULA_PROG)
    mv_ref.update(log2=16, padding=0)
    mv = build(COPULA_PROG)
    mv.update(log2=16, store_dir=str(tmp_path / 'bv'))
    assert mv.padding == 0
    assert abs(mv.deficit) < 1e-6
    m0 = mv.marginals[0]
    assert np.allclose(m0, mv_ref.marginals[0], atol=1e-12)
    mean0 = float(m0 @ mv.axis_xs[0])
    assert mean0 == pytest.approx(float(mv_ref.marginals[0] @ mv_ref.axis_xs[0]),
                                  rel=1e-12)


def test_update_store_dir_signed_axes(tmp_path):
    """Massive update on the signed (ssev) book: two-sided windows survive."""
    mv_ref = build(SIGNED_PROG)
    mv_ref.update(log2=16)
    mv = build(SIGNED_PROG)
    mv.update(log2=16, padding=1, store_dir=str(tmp_path / 'bv'))
    m1 = mv.marginals[1]
    assert np.allclose(m1, mv_ref.marginals[1], atol=1e-12)
    assert mv.axis_xs[1][0] < 0 < mv.axis_xs[1][-1]
    assert abs(mv.deficit) < 1e-6


def test_update_store_dir_discrete(tmp_path):
    """Massive update, discrete (dbvsev) mode: sparse lattice scatter."""
    mv_ref = build(DISCRETE_PROG)
    mv = build(DISCRETE_PROG)
    mv.update(padding=1, store_dir=str(tmp_path / 'bv'))
    assert np.allclose(mv.bivariate.density[:], mv_ref.density, atol=1e-12)
    assert np.allclose(mv.moments(3).to_numpy(),
                       mv_ref.moments(3).to_numpy(), rtol=1e-9)


def test_update_store_dir_netceded(tmp_path):
    """Massive update, netceded mode: sparse comonotone scatter + kernel."""
    mv_ref = build(f'netceded {NC_PROG}')
    mv = build(f'netceded {NC_PROG}')
    # padding=1 matches the in-core build_netceded_joint (agg.padding == 1)
    mv.update(padding=1, store_dir=str(tmp_path / 'bv'))
    assert np.allclose(mv.bivariate.density[:], mv_ref.density, atol=1e-12)
    m0, m1 = mv.marginals
    r0, r1 = mv_ref.marginals
    assert np.allclose(m0, r0, atol=1e-12)
    assert np.allclose(m1, r1, atol=1e-12)
    assert mv.summary_df is not None
    assert mv.info


def test_massive_container_probes(tmp_path):
    """marginal(i) / slice() / transformed_moments / moments(4) streamed."""
    mv_ref = build(COPULA_PROG)
    mv_ref.update(log2=16)
    mv = build(COPULA_PROG)
    mv.update(log2=16, padding=1, store_dir=str(tmp_path / 'bv'))
    bd = mv.bivariate
    # marginal(i): a GridDistribution with the exact accumulator mass
    g0 = bd.marginal(0)
    assert g0.mean() == pytest.approx(
        float(mv.marginals[0] @ mv.axis_xs[0]), rel=1e-12)
    # slice: conditional matches the normalised in-core row
    idx = int(np.argmax(mv_ref.marginals[0]))
    x = mv.axis_xs[0][idx]
    cond = bd.slice(x=x)
    row = mv_ref.density[idx, :]
    assert np.allclose(cond.p, row / row.sum(), atol=1e-12)
    # streamed transformed_moments == in-core transformed_moments
    f = lambda x, y: x + 0.5 * y
    tm_m = bd.transformed_moments(f)
    tm_r = mv_ref.bivariate.transformed_moments(f)
    assert tm_m['mean'] == pytest.approx(tm_r['mean'], rel=1e-10)
    assert tm_m['sd'] == pytest.approx(tm_r['sd'], rel=1e-8)
    # order-4 moments stream from disk and match the in-core frame
    assert np.allclose(bd.moments(4).to_numpy(),
                       mv_ref.bivariate.moments(4).to_numpy(), rtol=1e-8)


def test_massive_reopen_round_trip(tmp_path):
    """§8.6: reopen(store_dir) reproduces marginals, moments and the density."""
    from aggregate.bivariate import MassiveBivariateDistribution
    store = tmp_path / 'bv'
    mv = build(COPULA_PROG)
    mv.update(log2=16, store_dir=str(store))
    bd = mv.bivariate
    re = MassiveBivariateDistribution.reopen(str(store))
    assert np.array_equal(re.marg0, bd.marg0)
    assert np.array_equal(re.marg1, bd.marg1)
    assert np.array_equal(re.raw_moments, bd.raw_moments)
    assert re.bs0 == bd.bs0 and re.bs1 == bd.bs1
    assert re.total_mass == bd.total_mass
    assert re.axis_names == bd.axis_names
    assert re.meta['name'] == mv.name
    assert np.allclose(re.density[50:60, :], bd.density[50:60, :])


def test_massive_reopen_rejects_non_store(tmp_path):
    from aggregate.bivariate import MassiveBivariateDistribution
    with pytest.raises(FileNotFoundError, match='massive bivariate store'):
        MassiveBivariateDistribution.reopen(str(tmp_path))


def test_massive_deficit_guard_undersized_window(tmp_path):
    """§8.5: a deliberately undersized (pinned) window shows up in deficit.

    bs=50 (not 1): the FFT buffer must still reach physical 0..top, so a
    fine bs under a tiny window forces a full-size out-of-core buffer and
    the test costs ~20s for no extra coverage; at bs=50 the same clipped
    window trips the guard in under a second.
    """
    mv = build(COPULA_PROG)
    with pytest.warns():   # sizing warns about the uncovered window
        mv.update(log2=(6, 6), bs=(50, 50), store_dir=str(tmp_path / 'bv'))
    assert mv._clipped
    assert mv.deficit > 1e-3


def test_massive_density_df_raises(tmp_path):
    """density_df refuses a whole-store read with an actionable message."""
    mv = build(DISCRETE_PROG)
    mv.update(store_dir=str(tmp_path / 'bv'))
    with pytest.raises(ValueError, match='disk-backed'):
        mv.density_df


# ----------------------------------------------------------------------
# Phase 3 (§8.4): dict pushforward -- streamed, total, constants, audit
# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def massive_pair(tmp_path_factory):
    """One massive + one matched in-core update, shared across §8.4 tests."""
    mv_ref = build(COPULA_PROG)
    mv_ref.update(log2=16)
    mv = build(COPULA_PROG)
    store = tmp_path_factory.mktemp('pf') / 'bv'
    mv.update(log2=16, padding=1, store_dir=str(store))
    return mv, mv_ref


def test_pushforward_streamed_matches_incore(massive_pair):
    """§8.4: streamed dict pushforward == in-core pushforward, key by key."""
    mv, mv_ref = massive_pair
    fns = {'net': lambda x, y: x + 0.8 * y,
           'excess': lambda x, y: np.maximum(x + y - 1000.0, 0.0)}
    out = mv.bivariate.pushforward(fns, bs=25.0, bs_total=50.0)
    assert set(out) == {'net', 'excess', 'total'}
    for k, f in fns.items():
        gd = out[k]
        ref = mv_ref.bivariate.pushforward(
            f, bs=25.0, window=(float(gd.x[0]), float(gd.x[-1])))
        assert np.allclose(gd.p, ref.p, atol=1e-11), k
    # audit: Est vs EX means agree to discretisation noise (linear scatter
    # preserves the mean, so Err EX is fp-level)
    audit = out['net'].pushforward_audit_df
    assert set(audit.index) == {'net', 'excess', 'total'}
    assert (audit['Err EX'].abs() < 1e-10).all()


def test_pushforward_total_prebucket_linearity(massive_pair):
    """mean(total) == sum(mean(f_i)) -- the hard pre-bucket invariant."""
    mv, _ = massive_pair
    fns = {'a': lambda x, y: x, 'b': lambda x, y: 0.5 * y}
    out = mv.bivariate.pushforward(fns, bs=10.0, bs_total=10.0)
    assert out['total'].mean() == pytest.approx(
        out['a'].mean() + out['b'].mean(), rel=1e-12)
    # EX side too, from the shared audit frame
    audit = out['a'].pushforward_audit_df
    assert audit.loc['total', 'EX'] == pytest.approx(
        audit.loc['a', 'EX'] + audit.loc['b', 'EX'], rel=1e-12)


def test_pushforward_single_function_fast_path(massive_pair):
    """§8.4: one function => no 'total' key; bs_total raises."""
    mv, _ = massive_pair
    out = mv.bivariate.pushforward({'z': lambda x, y: x + y}, bs=25.0)
    assert set(out) == {'z'}
    with pytest.raises(ValueError, match='bs_total given with a single'):
        mv.bivariate.pushforward({'z': lambda x, y: x + y}, bs=25.0,
                                 bs_total=25.0)


def test_pushforward_missing_bs_total_raises(massive_pair):
    mv, _ = massive_pair
    with pytest.raises(ValueError, match='bs_total is required'):
        mv.bivariate.pushforward(
            {'a': lambda x, y: x, 'b': lambda x, y: y}, bs=10.0)


def test_pushforward_reserved_key_collision(massive_pair):
    mv, _ = massive_pair
    with pytest.raises(ValueError, match='reserved total key'):
        mv.bivariate.pushforward(
            {'total': lambda x, y: x, 'b': lambda x, y: y},
            bs=10.0, bs_total=10.0)


def test_pushforward_constant_short_circuit(massive_pair):
    """§8.4: constants (number and probe-constant callable) never enter the
    band loop; point mass at c with the stored total mass; correct scalar
    shift in the total."""
    mv, _ = massive_pair
    calls = {'n': 0}

    def premium(x, y):
        calls['n'] += 1
        return np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), 150e6)

    out = mv.bivariate.pushforward(
        {'loss': lambda x, y: x + y, 'premium': premium, 'fee': 7.5e6},
        bs=25.0, bs_total=25.0)
    # the probe evaluates the callable once; the band loop never does
    assert calls['n'] == 1
    for key, c in (('premium', 150e6), ('fee', 7.5e6)):
        gd = out[key]
        assert len(gd.x) == 1 and gd.x[0] == c
        assert gd.p[0] == pytest.approx(mv.bivariate.total_mass)
    # total = loss + 157.5e6 exactly
    assert out['total'].mean() == pytest.approx(
        out['loss'].mean() + (150e6 + 7.5e6) * mv.bivariate.total_mass,
        rel=1e-12)


def test_pushforward_pinned_window_clips_and_reports(massive_pair):
    """§8.4: values outside a pinned window clip to the edge buckets and are
    reported via .clipped_mass; total mass is retained."""
    mv, _ = massive_pair
    full = mv.bivariate.pushforward({'z': lambda x, y: x + y}, bs=25.0)['z']
    med = float(full.q(0.5))
    with pytest.warns(match='clipped'):
        out = mv.bivariate.pushforward({'z': lambda x, y: x + y}, bs=25.0,
                                       windows={'z': (0.0, med)})
    gd = out['z']
    assert gd.clipped_mass > 0.3          # ~half the mass lies above the median
    assert gd.p.sum() == pytest.approx(full.p.sum(), rel=1e-12)


def test_pushforward_incore_dict_dispatch_matches_massive(massive_pair):
    """API uniformity: BivariateDistribution.pushforward(dict) == massive."""
    mv, mv_ref = massive_pair
    fns = {'a': lambda x, y: x + 0.25 * y, 'b': lambda x, y: y}
    wins = {k: (0.0, 20000.0) for k in ('a', 'b', 'total')}
    got_m = mv.bivariate.pushforward(fns, bs=25.0, bs_total=50.0,
                                     windows=wins)
    got_i = mv_ref.bivariate.pushforward(fns, bs=25.0, bs_total=50.0,
                                         windows=wins)
    for k in ('a', 'b', 'total'):
        assert np.allclose(got_m[k].p, got_i[k].p, atol=1e-11), k
    # scalar (non-dict) form is unchanged and rejects the dict-only kwargs
    with pytest.raises(ValueError, match='dict'):
        mv_ref.bivariate.pushforward(lambda x, y: x + y, bs=25.0,
                                     bs_total=50.0)


# ----------------------------------------------------------------------
# Phase 4 (§7): decimation pyramid + Tier-1 static exhibit
# ----------------------------------------------------------------------

def test_pyramid_channels_exact(massive_pair):
    """L1 sum/max/min channels equal the 2x2 block reductions of the density."""
    mv, mv_ref = massive_pair
    pyr = mv.bivariate.pyramid
    assert pyr is not None
    levels = pyr.attrs['levels']
    assert levels, 'expected at least one pyramid level on a 512-wide grid'
    d = mv_ref.density                      # matched in-core density
    h, w = d.shape
    blocks = d.reshape(h // 2, 2, w // 2, 2)
    assert np.allclose(pyr['L1'][0], blocks.sum(axis=(1, 3)), atol=1e-7)
    assert np.allclose(pyr['L1'][1], blocks.max(axis=(1, 3)), atol=1e-12)
    assert np.allclose(pyr['L1'][2], blocks.min(axis=(1, 3)), atol=1e-12)


def test_pyramid_cascade_consistent(tmp_path):
    """Each level is the 2x2 reduction of the one below (multi-level store)."""
    mv = build(COPULA_PROG)
    mv.update(log2=20, store_dir=str(tmp_path / 'bv'))    # up to 1024-wide axes
    pyr = mv.bivariate.pyramid
    levels = pyr.attrs['levels']
    if len(levels) < 2:
        pytest.skip('grid too small for a second level')
    a1 = np.asarray(pyr['L1'][0])
    a2 = np.asarray(pyr['L2'][0])
    h, w = a1.shape
    assert np.allclose(a2, a1.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3)),
                       atol=1e-6)


def test_massive_plot_smoke(massive_pair):
    """Tier-1 exhibit renders: full view, zoom window, contours, exceedance."""
    mv, _ = massive_pair
    import matplotlib
    matplotlib.use('Agg', force=True)
    bd = mv.bivariate
    bd.plot()
    assert bd.figure is not None
    assert len(bd.figure.axes) >= 3
    # zoom to a quarter window at the same call cost
    x_mid = float(bd.xs0[len(bd.xs0) // 2])
    y_mid = float(bd.xs1[len(bd.xs1) // 2])
    bd.plot(window=((bd.xs0[0], x_mid), (bd.xs1[0], y_mid)),
            contours=True, exceedance=True, log=True)
    # the class stub delegates for a massive object
    mv.plot()
    import matplotlib.pyplot as plt
    plt.close('all')


def test_massive_plot_slice_smoke(massive_pair):
    mv, _ = massive_pair
    import matplotlib
    matplotlib.use('Agg', force=True)
    bd = mv.bivariate
    x = float(bd.xs0[int(np.argmax(bd.marg0))])
    ax = bd.plot_slice(x=x)
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close('all')


def test_pyramid_survives_reopen(massive_pair):
    from aggregate.bivariate import MassiveBivariateDistribution
    mv, _ = massive_pair
    re = MassiveBivariateDistribution.reopen(mv.bivariate.store_dir)
    assert re.pyramid is not None
    assert re.pyramid.attrs['levels'] == mv.bivariate.pyramid.attrs['levels']


# ----------------------------------------------------------------------
# Phase 5 (§7.3): Tier-2 interactive explorer (construction smoke only)
# ----------------------------------------------------------------------

def test_explore_builds_layout(massive_pair):
    """explore() composes the holoviews app (no server needed to construct)."""
    hv = pytest.importorskip('holoviews')
    pytest.importorskip('datashader')
    pytest.importorskip('bokeh')
    mv, _ = massive_pair
    app = mv.bivariate.explore(pixels=256)
    assert isinstance(app, hv.Layout)
    # main adjoint (image + marginals) plus slice and readout panels
    assert len(app) == 3


def test_kernel_staging_lifecycle(tmp_path):
    """z1/z2 staging is deleted by default and retained with keep_transform."""
    mv = build(DISCRETE_PROG)
    d1 = tmp_path / 'drop'
    massive_bivariate_convolution(**_copula_kernel_kwargs(mv, d1))
    assert (d1 / 'density.zarr').exists()
    assert not (d1 / 'z1.zarr').exists()
    assert not (d1 / 'z2.zarr').exists()
    d2 = tmp_path / 'keep'
    massive_bivariate_convolution(
        **_copula_kernel_kwargs(mv, d2, keep_transform=True))
    assert (d2 / 'z1.zarr').exists()
    assert (d2 / 'z2.zarr').exists()


# ----------------------------------------------------------------------
# [PnL-Generic-Final] massive-source group ledger: one band sweep
# ----------------------------------------------------------------------
def _tiny_massive(tmp_path):
    """A hand-built 4 x 3 disk-backed joint (atoms small enough to check by
    hand) plus its in-core twin for the cross-check."""
    from aggregate.bivariate import (BivariateDistribution,
                                     MassiveBivariateDistribution)
    xs0 = np.array([0.0, 10.0, 20.0, 30.0])          # loss L
    xs1 = np.array([0.0, 5.0, 10.0])                  # recovery R
    dens = np.array([[0.20, 0.05, 0.00],
                     [0.15, 0.10, 0.05],
                     [0.05, 0.10, 0.10],
                     [0.00, 0.05, 0.15]])
    store = tmp_path / 'pnl_bv'
    store.mkdir()
    z = zarr.open_array(str(store / 'density.zarr'), mode='w',
                        shape=dens.shape, chunks=(2, 3), dtype=float)
    z[:] = dens
    massive = MassiveBivariateDistribution(
        str(store), xs0, xs1, 10.0, 5.0, dens.sum(axis=1), dens.sum(axis=0),
        float(dens.sum()), 0.0, np.zeros((4, 4)), density=z)
    incore = BivariateDistribution(dens, xs0, xs1, bs_ceded=10.0, bs_net=5.0)
    return massive, incore


def _ledger_groups(bs=0.0):
    from aggregate import Leg, Group
    return [
        Group('gross', 'sell',
              [Leg('premium', 18.0, bs=bs)],
              [Leg('loss', lambda l: l, bs=bs)]),
        Group('occ xl', 'buy',
              [Leg('ceded premium', 4.0, bs=bs)],
              [Leg('recovery', lambda l, r: r, is2d=True, bs=bs)]),
    ]


def test_massive_pnl_one_sweep_ledger(tmp_path):
    """A zarr-backed joint evaluates the whole ledger in one band sweep:
    same row template as the in-memory route, mean(result) == sum of the
    signed leg means to VALIDATION_NOISE, and every leg audited."""
    from aggregate import PnL
    from aggregate.moments import VALIDATION_NOISE
    massive, incore = _tiny_massive(tmp_path)
    pm = PnL(name='massive', source=massive, groups=_ledger_groups(bs=1.0))
    pi = PnL(name='incore', source=incore, groups=_ledger_groups())
    # identical row template (the shared _ledger_plan) on sheet and card
    assert list(pm.stats_df.index) == list(pi.stats_df.index)
    assert list(pm.summary_df.index) == list(pi.summary_df.index)
    # the sweep's exact means match the in-core exact means, row by row
    for row in pm.stats_df.index:
        assert pm.stats_df.loc[row, 'EX'] == pytest.approx(
            pi.stats_df.loc[row, 'EX'], abs=1e-12), row
    # mean(result) == sum of signed leg means -- the derived row is pushed as
    # its own signed-sum function, never a sum of bucketed legs
    leg_means = sum(pm.stats_df.xs(r, level='Label')['EX'].iloc[0]
                    for r in ('premium', 'loss', 'ceded premium', 'recovery'))
    assert abs(pm.est_m - leg_means) < 10 * VALIDATION_NOISE
    # [Massive-Kappa-Second-Sweep]: the massive ladder stays MARGINAL (each
    # cell the row's own quantile), unlike the in-core scenario columns
    loss_gd = pm.density_df['loss']
    assert pm.stats_df.xs('loss', level='Label')['P50'].iloc[0] == \
        pytest.approx(float(loss_gd.q(0.5)))
    # the massive card carries full marginal percentiles (grand rows reused
    # from the sweep's own ledger rows -- no NaN holes)
    assert not pm.summary_df[['P01', 'Median', 'P99']].isna().any().any()
    # every declared leg is bs > 0 and audited (linear scheme: means match)
    v = pm.validation_df
    assert set(v.index) == {'premium', 'loss', 'ceded premium', 'recovery'}
    assert v['abs_err'].max() < 10 * VALIDATION_NOISE
    # accessors ride the grand result
    assert pm.prob_eq_0 == pytest.approx(pi.prob_eq_0, abs=1e-12)
    assert pm.q(0.5) == pytest.approx(pi.q(0.5), abs=1.0)  # bucketed grid


def test_massive_pnl_requires_bs_per_leg(tmp_path):
    from aggregate import PnL
    massive, _incore = _tiny_massive(tmp_path)
    with pytest.raises(ValueError, match='bs > 0'):
        PnL(name='bad', source=massive, groups=_ledger_groups(bs=0.0))
