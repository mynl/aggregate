"""Out-of-core compute core for the massive (disk-backed) bivariate compound.

The 2-D, disk-backed sibling of :mod:`aggregate._aggregate_compute`
(``dev/plan-bv.md``): the separable 2-D compound ``iFFT2(freq_pgf(FFT2(S)))``
run as three streamed passes over 2-D-tiled zarr stores, so the realized joint
density lives **on disk only** and peak RAM is bounded by a band regardless of
grid size. The corner-turn (the transpose between the two FFT directions) is
done *through disk*: pass 2 simply reads the pass-1 store column-wise -- tiles
in orthogonal order -- no in-memory shuffle, no dask.

1. **Severity lay-in + row-FFT -> ``z1``** (``M0 x nf1`` complex,
   ``nf1 = M1//2 + 1``). The per-claim severity ``S`` is tiny relative to the
   FFT buffer, so only its ~``nS0`` row bands are ever written; zarr leaves
   unwritten chunks at fill value 0, so ``z1`` is sparse on disk and this pass
   is near-free. A scipy.sparse ``S`` (the netceded comonotone scatter, one
   cell per gross grid point) streams row bands without ever densifying whole.
2. **Corner-turn: column bands of ``z1`` -> FFT(axis 0) -> pgf -> iFFT(axis 0)
   -> ``z2``** (dense complex; the expensive pass -- all the FFT work and a
   full write of the staging store).
3. **Inverse row-FFT + window relabel + fold-everything.** Iterate the
   *output* rows in bands (the window crop skips the ``M0 - N0`` unused buffer
   rows); output row ``r`` lives at buffer row ``(r + j0_0) mod M0``, at most
   two contiguous reads per band. After the ``irfft`` the axis-1 window is an
   in-RAM roll. Each finished band is folded in one sweep: written to
   ``density.zarr`` (float64, physical order -- post-hoc consumers never see
   ``j0`` again), marginals, total mass, mixed raw moments
   ``E[X^a Y^b], a, b <= 3``, and any caller hook (``on_band``).

The streamed relabel is exact: ``np.roll`` of a finished array is a cyclic
*renaming* -- buffer index ``i`` holds physical value
``x_min + ((i - j0) mod M) * bs`` -- and pass-3's reductions bin by value, so
streaming bands in buffer order with rolled labels equals rolling the array
(the 1-D argument of :func:`~aggregate._aggregate_compute.freq_sev_convolution`
lifted to 2-D). Exact when the aggregate support width ``< M*bs`` per axis;
``deficit`` reports what wrapped.

A leaf: numpy / scipy only, ``zarr`` imported lazily inside the entry points
(optional extra ``aggregate[massive]``); never imports ``_aggregate`` or
``bivariate``. Techniques proven in the sibling exploration repo bigbiv
(``T:/ai/big2conv``): bit-identical to the in-core engine at all chunk shapes,
peak RSS constant in grid size.
"""

import os
import shutil
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
import scipy.sparse as ssp

_MAX_MIXED_ORDER = 3       # mixed raw moments E[X^a Y^b] with a, b <= 3
_ZERO_SMALL = 1e-15        # clamp threshold, matching the in-core update_work


def _require_zarr():
    """Import and return ``zarr`` with an actionable error if missing."""
    try:
        import zarr
    except ImportError as e:
        raise ImportError(
            "the disk-backed (massive) bivariate update requires 'zarr'; "
            "install the optional extra: pip install aggregate[massive]"
        ) from e
    return zarr


@dataclass
class MassiveResult:
    """Handles and accumulators from a massive bivariate convolution.

    Attributes
    ----------
    store_dir : str
        The backing directory holding ``density.zarr`` (and, with
        ``keep_transform=True``, the ``z1.zarr`` / ``z2.zarr`` staging).
    density : zarr.Array
        The realized joint density, shape ``(N0, N1)`` float64, in physical
        (monotone-label) order on disk. Lazy -- slicing reads tiles.
    xs0, xs1 : ndarray
        Per-axis output label grids.
    bs0, bs1 : float
        Per-axis bucket sizes.
    marg0, marg1 : ndarray
        Exact axis marginals (folded from the density bands).
    total_mass : float
        Sum of the joint density (``1 - deficit``).
    deficit : float
        Mass wrapped / lost beyond the window.
    raw_moments : ndarray
        ``(4, 4)`` unnormalised mixed raw moment sums
        ``sum p * x^a * y^b`` for ``a, b <= 3``; entry ``[0, 0]`` is
        ``total_mass``.
    """

    store_dir: str
    density: object
    xs0: np.ndarray
    xs1: np.ndarray
    bs0: float
    bs1: float
    marg0: np.ndarray
    marg1: np.ndarray
    total_mass: float
    deficit: float
    raw_moments: np.ndarray


def _mixed_moment_band(xs0_band, band, ypow):
    """``(4, 4)`` partial mixed raw moment sums of one density band.

    ``band`` is ``(rc, N1)`` mass with row labels ``xs0_band`` and column-power
    matrix ``ypow[b] = xs1**b``; returns ``mom[a, b] = sum band * x^a * y^b``
    over the band (float64, numpy pairwise summation within each product).
    """
    band_y = band @ ypow.T                          # (rc, 4)
    mom = np.empty((_MAX_MIXED_ORDER + 1,) * 2)
    xa = np.ones_like(xs0_band)
    for a in range(_MAX_MIXED_ORDER + 1):
        mom[a, :] = xa @ band_y
        xa = xa * xs0_band
    return mom


# ----------------------------------------------------------------------
# visualization pyramid (plan-bv §7.1)
# ----------------------------------------------------------------------

_PYRAMID_MIN_DIM = 256      # coarsest level: both dims <= this


def _reduce_2x2(a, op):
    """2x2 block reduction of a 2-D array (both dims even) by ``op``."""
    h, w = a.shape
    b = a.reshape(h // 2, 2, w // 2, 2)
    if op == 'sum':
        return b.sum(axis=(1, 3))
    if op == 'max':
        return b.max(axis=(1, 3))
    return b.min(axis=(1, 3))


class PyramidBuilder:
    """Streamed sum/max/min decimation pyramid over the joint density.

    A map-tile pyramid (``pyramid.zarr`` beside the density): level ``k`` is
    a ``2^k x 2^k`` block reduction of the base grid, down to
    ``<= _PYRAMID_MIN_DIM`` per axis, with **three channels** per level
    because one reduction lies (plan-bv §7.1):

    - ``sum`` -- probability aggregates additively: a block-sum *is* the pmf
      on the coarser grid (the "body" channel a heatmap shows);
    - ``max`` -- preserves what sum washes out at coarse zoom: one-cell-wide
      filaments (the netceded comonotone curve), atoms, attachment kinks;
    - ``min`` -- reveals interior holes / exact-zero regions inside
      apparently solid mass.

    The base level (``L1``) folds from the pass-3 bands via :meth:`on_band`
    (bands are row-aligned to even offsets, so 2x2 blocks never straddle a
    band); higher levels cascade level-to-level in :meth:`finish`, streamed
    in row bands so RAM stays banded. float32 throughout (viz-only).

    Grids with both dims already ``<= _PYRAMID_MIN_DIM`` get an empty
    pyramid (attrs only) -- the plot reads the density directly there.
    """

    def __init__(self, store_dir, N0, N1, *, row_chunk=512, col_chunk=512,
                 min_dim=_PYRAMID_MIN_DIM):
        zarr = _require_zarr()
        self.path = os.path.join(store_dir, 'pyramid.zarr')
        self.group = zarr.open_group(self.path, mode='w')
        self.N0, self.N1 = int(N0), int(N1)
        self.row_chunk, self.col_chunk = int(row_chunk), int(col_chunk)
        self.min_dim = int(min_dim)
        self._levels = []           # [(name, n0, n1), ...]
        if max(self.N0, self.N1) > self.min_dim \
                and self.N0 % 2 == 0 and self.N1 % 2 == 0:
            self._l1 = self.group.create_array(
                'L1', shape=(3, self.N0 // 2, self.N1 // 2),
                chunks=(1, self.row_chunk, self.col_chunk), dtype='float32')
            self._levels.append(('L1', self.N0 // 2, self.N1 // 2))
        else:
            self._l1 = None

    def on_band(self, r0, r1, band):
        """Fold one pass-3 density band into the base pyramid level."""
        if self._l1 is None:
            return
        if r0 % 2 or (r1 - r0) % 2:
            # bands are row_chunk-aligned on power-of-two grids, so this
            # cannot happen in practice; skip rather than corrupt a block.
            return
        p0, p1 = r0 // 2, r1 // 2
        self._l1[0, p0:p1, :] = _reduce_2x2(band, 'sum').astype(np.float32)
        self._l1[1, p0:p1, :] = _reduce_2x2(band, 'max').astype(np.float32)
        self._l1[2, p0:p1, :] = _reduce_2x2(band, 'min').astype(np.float32)

    def finish(self):
        """Cascade the higher levels and stamp the group attrs.

        Each level reduces the previous one channel-by-channel (sum of sums,
        max of maxes, min of mins -- all composable), read in even-aligned
        row bands so nothing large is ever in RAM.
        """
        ops = ('sum', 'max', 'min')
        while self._levels:
            name, n0, n1 = self._levels[-1]
            if max(n0, n1) <= self.min_dim or n0 % 2 or n1 % 2:
                break
            prev = self.group[name]
            k = int(name[1:]) + 1
            nxt = self.group.create_array(
                f'L{k}', shape=(3, n0 // 2, n1 // 2),
                chunks=(1, self.row_chunk, self.col_chunk), dtype='float32')
            rc = max(2, self.row_chunk - self.row_chunk % 2)
            for ch, op in enumerate(ops):
                for r0 in range(0, n0, rc):
                    r1 = min(r0 + rc, n0)
                    block = np.asarray(prev[ch, r0:r1, :])
                    nxt[ch, r0 // 2:r1 // 2, :] = _reduce_2x2(
                        block, op).astype(np.float32)
            self._levels.append((f'L{k}', n0 // 2, n1 // 2))
        self.group.attrs['levels'] = [
            {'name': n, 'shape': [a, b]} for n, a, b in self._levels]
        self.group.attrs['base_shape'] = [self.N0, self.N1]
        self.group.attrs['channels'] = list(ops)
        return self.group


def massive_bivariate_convolution(S, freq_pgf, en, *,
                                  N0, N1, bs0, bs1,
                                  i0=(0, 0), j0=(0, 0), mlog2,
                                  store_dir,
                                  xs0=None, xs1=None,
                                  row_chunk=512, col_chunk=512,
                                  keep_transform=False,
                                  on_band=None, progress=None):
    """Out-of-core 2-D compound: ``iFFT2(freq_pgf(en, FFT2(S)))``, disk-backed.

    The massive sibling of the in-core
    :meth:`aggregate.bivariate.BivariateAggregate.update_work` FFT block and
    the 2-D lift of
    :func:`~aggregate._aggregate_compute.freq_sev_convolution` -- same
    ``i0`` / ``j0`` windowing vocabulary, same results to ``1e-12``, but the
    joint density never materialises in RAM (see the module docstring for the
    three passes).

    Parameters
    ----------
    S : ndarray or scipy.sparse matrix
        The per-claim bivariate severity pmf. Dense for the copula / discrete
        modes (tiny relative to the aggregate support); scipy.sparse (CSR
        recommended) for a scatter like the netceded comonotone curve whose
        dense form would not fit in RAM. ``S[a, b]`` is the mass at physical
        ``((a - i0[0]) * bs0, (b - i0[1]) * bs1)``.
    freq_pgf : callable
        The shared frequency PGF ``freq_pgf(en, z) -> ndarray``, elementwise
        in ``z`` (e.g. ``Frequency.freq_pgf``). Applied to raveled blocks (the
        empirical pgf assumes a 1-D argument).
    en : float
        Expected outer event count. ``en == 0`` short-circuits to a point mass
        at physical ``(0, 0)``.
    N0, N1 : int
        Output grid lengths (the measured window).
    bs0, bs1 : float
        Per-axis bucket sizes.
    i0 : (int, int), optional
        Per-axis severity buckets below physical 0 (signed lay-in wrap).
    j0 : (int, int), optional
        Per-axis output-window origin in buckets (``round(x_min / bs)``; may
        be negative on a signed axis).
    mlog2 : (int, int)
        Per-axis log2 FFT buffer length (decoupled from the output length so
        a tight window far from 0 does not alias -- from the update sizing).
    store_dir : str
        Backing directory (required -- this is the point). Created if absent;
        ``density.zarr`` and the staging stores live here.
    xs0, xs1 : ndarray, optional
        Output label grids. Default ``(j0 + arange(N)) * bs``; pass the bv's
        measured ``axis_xs`` so streamed moments match the in-core labels
        exactly.
    row_chunk, col_chunk : int, optional
        Band / tile sizes. Peak RAM ~ ``max(row_chunk * M1, M0 * col_chunk) *
        16`` bytes -- constant in grid size.
    keep_transform : bool, optional
        Keep the ``z1`` / ``z2`` staging stores after completion (default
        deletes them; ``z2`` is the largest artifact).
    on_band : callable, optional
        ``on_band(r0, r1, band)`` called once per finished pass-3 band (the
        ``(r1 - r0, N1)`` float64 density rows in physical order) -- the hook
        for the visualization pyramid and update-time pushforwards. Bands with
        no mass may be skipped in the degenerate ``en == 0`` case.
    progress : callable, optional
        ``progress(pass_no, frac)`` callback for long runs.

    Returns
    -------
    MassiveResult
        Handles + accumulators; the density stays on disk.
    """
    zarr = _require_zarr()

    N0, N1 = int(N0), int(N1)
    bs0, bs1 = float(bs0), float(bs1)
    i0_0, i0_1 = int(i0[0]), int(i0[1])
    j0_0, j0_1 = int(j0[0]), int(j0[1])
    M0, M1 = 1 << int(mlog2[0]), 1 << int(mlog2[1])
    nf1 = M1 // 2 + 1
    if xs0 is None:
        xs0 = (j0_0 + np.arange(N0, dtype=float)) * bs0
    else:
        xs0 = np.asarray(xs0, dtype=float)
    if xs1 is None:
        xs1 = (j0_1 + np.arange(N1, dtype=float)) * bs1
    else:
        xs1 = np.asarray(xs1, dtype=float)

    os.makedirs(store_dir, exist_ok=True)
    p1 = os.path.join(store_dir, 'z1.zarr')
    p2 = os.path.join(store_dir, 'z2.zarr')
    pd_ = os.path.join(store_dir, 'density.zarr')
    density = zarr.open(pd_, mode='w', shape=(N0, N1),
                        chunks=(row_chunk, col_chunk), dtype=np.float64)

    # ------------------------------------------------------------------
    # degenerate zero-risk book: point mass at physical (0, 0), placed at
    # output index (-j0) mod N per axis -- mirrors update_work's en == 0.
    # ------------------------------------------------------------------
    if en == 0:
        r = (-j0_0) % N0
        c = (-j0_1) % N1
        density[r, c] = 1.0
        marg0 = np.zeros(N0)
        marg1 = np.zeros(N1)
        marg0[r] = 1.0
        marg1[c] = 1.0
        mom = np.array([[xs0[r] ** a * xs1[c] ** b
                         for b in range(_MAX_MIXED_ORDER + 1)]
                        for a in range(_MAX_MIXED_ORDER + 1)])
        if on_band is not None:
            r0 = (r // row_chunk) * row_chunk
            r1 = min(r0 + row_chunk, N0)
            band = np.zeros((r1 - r0, N1))
            band[r - r0, c] = 1.0
            on_band(r0, r1, band)
        return MassiveResult(store_dir, density, xs0, xs1, bs0, bs1,
                             marg0, marg1, 1.0, 0.0, mom)

    sparse = ssp.issparse(S)
    if sparse:
        S = S.tocsr()
    else:
        S = np.asarray(S, dtype=float)
    nS0, nS1 = S.shape

    z1 = zarr.open(p1, mode='w', shape=(M0, nf1),
                   chunks=(row_chunk, col_chunk), dtype=np.complex128)
    z2 = zarr.open(p2, mode='w', shape=(M0, nf1),
                   chunks=(row_chunk, col_chunk), dtype=np.complex128)

    # ------------------------------------------------------------------
    # Pass 1 -- severity lay-in + row-FFT -> z1 (sparse on disk).
    # Two row segments (the 2-D lift of the 1-D signed lay-in): S rows
    # [i0_0, nS0) land at buffer rows [0, nS0 - i0_0); the i0_0 negative rows
    # wrap to the top, buffer rows [M0 - i0_0, M0).
    # ------------------------------------------------------------------
    segments = [(i0_0, nS0, 0)]                 # (S row start, stop, buffer row)
    if i0_0:
        segments.append((0, i0_0, M0 - i0_0))
    n_rows = nS0
    done = 0
    for s0, s1, b0 in segments:
        for r0 in range(s0, s1, row_chunk):
            r1 = min(r0 + row_chunk, s1)
            block = S[r0:r1]
            if sparse:
                if block.nnz == 0:              # zarr fill 0 stands in
                    done += r1 - r0
                    continue
                block = block.toarray()
            buf = np.zeros((r1 - r0, M1))
            # axis-1 signed lay-in: physical 0 at column 0, i0_1 negative
            # buckets wrapped to the top of the period M1.
            buf[:, :nS1 - i0_1] = block[:, i0_1:]
            if i0_1:
                buf[:, M1 - i0_1:] = block[:, :i0_1]
            z1[b0 + (r0 - s0):b0 + (r1 - s0), :] = sfft.rfft(buf, axis=1)
            done += r1 - r0
            if progress is not None:
                progress(1, done / n_rows)

    # ------------------------------------------------------------------
    # Pass 2 -- the corner-turn: column bands of z1, FFT down axis 0, the
    # (elementwise) pgf, inverse FFT down axis 0 -> z2. Reading z1 column-wise
    # IS the transpose -- tiles in orthogonal order; unwritten chunks
    # materialise as zeros for free.
    # ------------------------------------------------------------------
    for c0 in range(0, nf1, col_chunk):
        c1 = min(c0 + col_chunk, nf1)
        col = np.asarray(z1[:, c0:c1])
        F = sfft.fft(col, axis=0)
        # freq_pgf is mathematically elementwise but the empirical
        # implementation assumes a 1-D argument -- ravel, apply, reshape
        # (same as the in-core update_work).
        F = np.asarray(freq_pgf(en, F.ravel())).reshape(F.shape)
        z2[:, c0:c1] = sfft.ifft(F, axis=0)
        if progress is not None:
            progress(2, c1 / nf1)

    # ------------------------------------------------------------------
    # Pass 3 -- inverse row-FFT + window relabel + fold everything. Iterate
    # the OUTPUT rows (the crop skips (M0 - N0)/M0 of the read); output row r
    # lives at buffer row (r + j0_0) mod M0 -> at most 2 contiguous reads.
    # ------------------------------------------------------------------
    marg0 = np.zeros(N0)
    marg1 = np.zeros(N1)
    total_parts = []
    mom_parts = []
    ypow = np.vstack([xs1 ** b for b in range(_MAX_MIXED_ORDER + 1)])
    for r0 in range(0, N0, row_chunk):
        r1 = min(r0 + row_chunk, N0)
        rc = r1 - r0
        b0 = (r0 + j0_0) % M0
        if b0 + rc <= M0:
            block = np.asarray(z2[b0:b0 + rc, :])
        else:
            block = np.vstack([np.asarray(z2[b0:M0, :]),
                               np.asarray(z2[0:b0 + rc - M0, :])])
        band = sfft.irfft(block, n=M1, axis=1)
        band = np.real(band)
        # axis-1 window: cyclic relabel in RAM, then crop (see module notes).
        if j0_1:
            band = np.roll(band, -j0_1, axis=1)
        band = np.ascontiguousarray(band[:, :N1])
        band[np.abs(band) < _ZERO_SMALL] = 0.0
        density[r0:r1, :] = band
        marg0[r0:r1] = band.sum(axis=1)
        marg1 += band.sum(axis=0)
        total_parts.append(band.sum())
        mom_parts.append(_mixed_moment_band(xs0[r0:r1], band, ypow))
        if on_band is not None:
            on_band(r0, r1, band)
        if progress is not None:
            progress(3, r1 / N0)

    total = float(np.sum(total_parts))
    raw_moments = np.sum(np.stack(mom_parts), axis=0)
    if not keep_transform:
        del z1, z2
        shutil.rmtree(p1, ignore_errors=True)
        shutil.rmtree(p2, ignore_errors=True)

    return MassiveResult(store_dir, density, xs0, xs1, bs0, bs1,
                         marg0, marg1, total, float(1.0 - total), raw_moments)
