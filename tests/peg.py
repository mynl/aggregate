"""PEG — the two-unit Portfolio regression fixture for the refactor.

Single helper, ``build_peg``, that constructs and (optionally) updates and
calibrates the canonical regression Portfolio. Reused by

- ``tests/capture_peg_baseline.py`` (one-shot to write the JSON baseline)
- ``tests/test_portfolio_peg_regression.py`` (the regression assertions)

The Portfolio exercises a limit-and-attachment severity, a three-component
severity mixture per unit, and gamma frequency mixing with different mixing
CVs per unit (the standard mixed-vs-independent correlation case). Five
calibrated distortions follow (``ccoc``, ``ph``, ``wang``, ``dual``,
``tvar``). Small enough to run in a few seconds, large enough to exercise
the full calibrate → analyze pipeline.

The DecL program here is the contract. Do not change it without re-capturing
the baseline.
"""
from __future__ import annotations

from aggregate import build


PEG_PROGRAM = (
    'port PEG '
    'agg A 100 claims 50 xs 0 sev lognorm [30 50 100] cv [0.1 .6 1.8] '
    '       wts [.5 .25 .25] mixed gamma .25 '
    'agg B 150 claims 100 xs 0 sev lognorm [30 50 100] cv [0.1 .6 1.8] '
    '       wts [.5 .25 .25] mixed gamma .35'
)


def build_peg(*, update=True, calibrate=True, p=0.995, coc=0.15,
              log2=16, bs=None):
    """Construct (and optionally update + calibrate) the PEG Portfolio.

    Parameters
    ----------
    update : bool
        If True, call ``port.update(log2, bs)`` after construction.
    calibrate : bool
        If True (and ``update`` is True), call
        ``port.calibrate_distortions(COCs=[coc], Ps=[p])`` so
        ``port.dists`` is populated.
    p : float
        Calibration percentile (default 0.995).
    coc : float
        Cost-of-capital target for calibration (default 0.15).
    log2 : int
        FFT grid size = ``2**log2`` (default 16).
    bs : float or None
        Bucket size. If None, picked via ``port.best_bucket(log2)``.

    Returns
    -------
    Portfolio
        Configured per the flags above.

    Notes
    -----
    Sub-project D renames the calibration API; when that lands, update the
    ``calibrate_distortions`` call here to the new signature. The baseline
    JSON values are unchanged — same numerics, different access pattern.
    """
    port = build(PEG_PROGRAM)
    if update:
        if bs is None:
            bs = port.best_bucket(log2)
        port.update(log2=log2, bs=bs, remove_fuzz=True)
        if calibrate:
            port.calibrate_distortions(COCs=[coc], Ps=[p])
    return port
