"""Pure-function compute core for the aggregate FFT convolution.

Extracted from ``Aggregate`` (Phase 2A) so the FFT-PGF-iFFT kernel is reachable
and testable **without** a full :meth:`Aggregate.update` -- it can be driven
directly with hand-built severity vectors and a frequency PGF, cross-checked
against ``np.convolve`` (fixed frequency) or a direct Panjer/compound sum, and
inspected in a notebook. ``Aggregate`` keeps thin methods that call in here.

A leaf: numpy / scipy.fft / the ``utilities`` FFT helpers only; it never imports
``_aggregate``.
"""

import numpy as np
import scipy.fft as sfft

from .utilities import ft, ift


def freq_sev_convolution(sev_density, freq_pgf, n, *, N, bs, i0=0, x_min=0.0,
                         en=None, freq_name='', padding=1):
    """Aggregate density by FFT convolution: ``iFFT(freq_pgf(n, FFT(sev)))``.

    The single source of truth for the FFT-PGF-iFFT core (Mildenhall 2024,
    §2.2). The compound (aggregate) law of ``S = X_1 + ... + X_K`` with random
    claim count ``K`` is ``iFFT( P_K( FFT(sev) ) )`` where ``P_K`` is the
    frequency probability generating function -- valid because the PGF acts
    elementwise on the transformed severity. This function is what
    :meth:`Aggregate._fft_aggregate` delegates to; it takes plain arrays and a
    PGF callable, so it can be exercised directly (see
    ``tests/test_aggregate_compute.py``).

    Parameters
    ----------
    sev_density : np.ndarray
        Discretised severity, length ``N``; ``sev_density[j]`` is the mass at
        physical ``(j - i0) * bs``.
    freq_pgf : callable
        The frequency PGF ``freq_pgf(n, z) -> ndarray``, elementwise in the
        transformed argument ``z`` (e.g. ``Frequency.freq_pgf``).
    n : int
        Grid length passed to the PGF (``self.n``; the FFT length parameter).
    N : int
        Length of the output grid (``len(self.xs)``).
    bs : float
        Bucket size; only used to locate the output-window origin.
    i0 : int, optional
        Number of severity buckets below physical 0 (signed severity). ``0`` on
        the default non-negative grid.
    x_min : float, optional
        Output-window origin; ``0.0`` on the default zero-based grid.
    en : array-like, optional
        Expected claim counts; only consulted for the fixed-frequency,
        single-claim shortcut. ``None`` skips the shortcut.
    freq_name : str, optional
        Frequency kind; the fixed-1 shortcut fires only for ``'fixed'``.
    padding : int, optional
        FFT padding factor passed to :func:`~aggregate.utilities.ft` /
        :func:`~aggregate.utilities.ift`. Default 1.

    Returns
    -------
    agg_density : np.ndarray
        Aggregate density on the output grid (length ``N``).
    ftagg_density : np.ndarray
        FT of the aggregate (padded length); callers that don't need it discard.

    Notes
    -----
    Two paths, selected by whether any offset is active:

    - **Default (``i0 == 0`` and window origin ``x_min == 0``).** The original
      ``ft`` / ``freq_pgf`` / ``ift`` path, byte-for-byte identical to prior
      releases.
    - **Signed / windowed.** Negatives live at the top of the padded length-``M
      = N << padding`` FFT buffer (period ``M*bs``); the severity is laid in with
      physical 0 at index 0 (``i0`` negative buckets wrapped to the top). After
      the FFT the result is *relabelled* onto the output window by a single
      ``np.roll`` of ``-round(x_min/bs)`` and the first ``N`` buckets kept.
      Relabelling a finished, exact array carries no ``N·s`` shift term, so this
      is correct for random as well as fixed frequency. Exact when the aggregate
      support width ``W < M*bs``; a narrower window shows a two-sided deficit.
    """
    j0 = int(round(x_min / bs)) if bs else 0
    if i0 == 0 and j0 == 0:
        # ---- default non-negative, zero-based path (unchanged) ----
        if n == 0:
            out = np.zeros(N)
            out[0] = 1.0
            return out, ft(out, padding)
        z = ft(sev_density, padding)
        ftagg = freq_pgf(n, z)
        if en is not None and np.sum(en) == 1 and freq_name == 'fixed':
            return sev_density.copy(), ftagg
        return np.real(ift(ftagg, padding)), ftagg

    # ---- signed / windowed path (F1 negative-x + F2 output window) ----
    M = N << padding
    if n == 0:
        # Zero-risk: point mass at physical 0, placed at FFT index 0 so the
        # output-window roll below sends it to the correct output bucket.
        a = np.zeros(M)
        a[0] = 1.0
        ftagg = sfft.rfft(a)
    else:
        # Lay the severity into the length-M buffer: physical 0..(N-1-i0)*bs at
        # indices 0..N-1-i0; the i0 negative buckets wrap to the very top of M
        # (indices M-i0..M-1). Equivalent to np.roll(sev, -i0) but into the
        # padded length so the period is M*bs, not N*bs.
        g = np.zeros(M)
        g[:N - i0] = sev_density[i0:]
        if i0:
            g[M - i0:] = sev_density[:i0]
        z = sfft.rfft(g)
        ftagg = freq_pgf(n, z)
        a = sfft.irfft(ftagg, M)
    # F2: relabel onto the output window. Roll so x_min lands at output index 0
    # (j0 may be negative when x_min < 0), then keep the first N.
    agg = np.roll(a, -j0)[:N]
    return agg, ftagg
