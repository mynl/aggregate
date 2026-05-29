"""Before/after baseline corpus for the v1.0 core-compute refactor.

The single source of truth for the harness DecL specs, pinned grids, fixed
distortions, asset levels, and what to snapshot. ``capture.py`` reads from
here to write ``data/<case>__<frame>.parquet`` + ``data/manifest.json``;
``test_baseline.py`` reads the same dicts to rebuild each case and compare.

See ``dev/plan-baseline-harness.md`` for the design.

Conventions
-----------
* Grids are pinned per case (no recommender — it may itself be refactored).
* ``bs`` is always an exact binary fraction (the ``round_bucket`` invariant).
* No RNG anywhere. No ``calibrate_distortions`` (Newton solve avoided).
* Distortions are fixed-shape: ``ccoc 0.10`` (mass at 0), ``dual 1.85``,
  ``tvar 0.65`` (neither has a mass).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Aggregate cases
# ---------------------------------------------------------------------------
# Each entry: (program, grid).  grid is the kwargs forwarded to ``update``:
#   log2, bs, padding, normalize.

AGG_CASES = {
    # 1a/1b: fixed-1 shortcut vs full-FFT path — must agree to ~eps.
    "Base.FixedOne": (
        "agg Base.FixedOne 1 claim sev lognorm 100 cv 0.5 fixed",
        dict(log2=16, bs=1/32, padding=1, normalize=True),
    ),
    "Base.DfreqOne": (
        "agg Base.DfreqOne dfreq [1] sev lognorm 100 cv 0.5",
        dict(log2=16, bs=1/32, padding=1, normalize=True),
    ),
    # 2: symmetric discrete — zero skew, fuzz-sensitive (validation regression).
    "Sym.Dice": (
        "agg Sym.Dice dfreq [1] dsev [1:6]",
        dict(log2=5, bs=1, padding=1, normalize=True),
    ),
    # 3: thick tail — exercises tail accuracy / wide grid.
    "Tail.LN": (
        "agg Tail.LN 5 claims sev lognorm 100 cv 3 poisson",
        dict(log2=16, bs=5, padding=2, normalize=True),
    ),
    # 4: occurrence + aggregate reinsurance — the subject/after reporting case.
    "Re.Both": (
        "agg Re.Both 100 claims 5000 xs 0 sev lognorm 50 cv 1.5 "
        "occurrence net of 3500 po 4000 xs 1000 poisson "
        "aggregate net of 2000 xs 3000",
        dict(log2=16, bs=1/4, padding=2, normalize=True),
    ),
    # 5: defective + no-2nd-moment — pareto α=1.5, normalize=False so sum(p) < 1.
    #    Exercises the defective path (forwards/backwards S diverge by the
    #    deficit) and the no-finite-2nd-moment path.
    "Def.Pareto": (
        "agg Def.Pareto 1 claim sev 1000 * pareto 1.5 - 1000 fixed",
        dict(log2=16, bs=3125/8192, padding=1, normalize=False),
    ),
    # 6: bounded mixture.
    "Mixture": (
        "agg Mix [10 20] claims [100 400] xs 0 "
        "sev lognorm [20 40 100] cv [.5 .6 .7] wts=3 "
        "mixed gamma .2",
        dict(log2=16, bs=1/8, padding=1, normalize=True),
    ),
}


# ---------------------------------------------------------------------------
# Portfolio cases
# ---------------------------------------------------------------------------
# Each entry: (program, grid, allocation_methods).
#   grid kwargs forwarded to Portfolio.update: log2, bs, padding.
#   allocation_methods: which methods to snapshot for the no-mass distortions.
#                       CCoC (mass) is restricted per case (see comments).

PORT_CASES = {
    # P1: unbounded (gamma/lognorm, fixed-1) — PIR CNC.
    "Port.CNC": (
        "port Port.CNC "
        "agg CNC.NonCat 25 claim sev gamma   80 cv 0.15 mixed gamma .2 "
        "agg CNC.Cat    5  claim 200 xs 0 sev lognorm 40 cv 1.50 mixed ig .2",
        dict(log2=16, bs=1/4, padding=1),
    ),
    # P2: bounded beta mixture.
    "Port.Bounded": (
        "port Port.Bounded "
        "agg A dfreq[1:3] sev 500 * beta 5 2 "
        "agg B dfreq[1:5] sev 800 * beta 7 2",
        dict(log2=16, bs=1/8, padding=0),
    ),
    # P3: bounded discrete (max 199, fixed-1) — Bodoff.
    "Port.Bodoff": (
        "port Port.Bodoff "
        "agg wind  1 claim sev dhistogram xps [0,  99] [0.80, 0.20] fixed "
        "agg quake 1 claim sev dhistogram xps [0, 100] [0.95, 0.05] fixed",
        dict(log2=8, bs=1, padding=1),
    ),
}


# ---------------------------------------------------------------------------
# Per-(case, distortion) allocation methods to snapshot
# ---------------------------------------------------------------------------
# ``lifted`` is unstable for a mass distortion (CCoC) on an unbounded support,
# which is exactly what meta.6 / Portfolio D2 will refuse. Until then we still
# capture what the code currently produces for the *non-mass* distortions on
# Port.CNC under lifted (they are fine) but skip CCoC+lifted on Port.CNC to
# keep the baseline meaningful. Bounded ports take both methods for every
# distortion.

PORT_METHODS = {
    ("Port.CNC", "ccoc"):     ["linear"],
    ("Port.CNC", "dual"):     ["linear", "lifted"],
    ("Port.CNC", "tvar"):     ["linear", "lifted"],
    ("Port.Bounded", "ccoc"): ["linear", "lifted"],
    ("Port.Bounded", "dual"): ["linear", "lifted"],
    ("Port.Bounded", "tvar"): ["linear", "lifted"],
    ("Port.Bodoff", "ccoc"):  ["linear", "lifted"],
    ("Port.Bodoff", "dual"):  ["linear", "lifted"],
    ("Port.Bodoff", "tvar"):  ["linear", "lifted"],
}


# ---------------------------------------------------------------------------
# Fixed distortions (no Newton calibration in the baseline)
# ---------------------------------------------------------------------------
# (name, kind, shape).  ``mass`` flags the only one with a survival-function
# jump at s=0 (CCoC). The lifted-on-unbounded+mass refusal applies only to
# ``ccoc`` paired with Port.CNC under ``allocation='lifted'``.

DISTORTIONS = [
    # Each entry: kwargs to ``Distortion(name=..., **kwargs)`` plus a
    # ``mass`` flag and a stable ``label`` used to name snapshot files.
    dict(label="ccoc", kind="ccoc", kwargs=dict(r=0.10),       mass=True),
    dict(label="dual", kind="dual", kwargs=dict(shape=1.85),   mass=False),
    dict(label="tvar", kind="tvar", kwargs=dict(shape=0.65),   mass=False),
]


# ---------------------------------------------------------------------------
# Scalar readouts (asset levels for q / tvar)
# ---------------------------------------------------------------------------
SCALAR_PS = (0.9, 0.99, 0.999)
TVAR_PS = (0.99,)
PRICING_P = 0.99


# ---------------------------------------------------------------------------
# What to snapshot per case
# ---------------------------------------------------------------------------
# Frame -> column filter.  ``None`` means "all columns".
# ``density_df`` is filtered to keep the snapshot small: probabilities and
# conditional-expectation columns plus loss / S.

AGG_FRAMES = {
    "stats_df": None,
    "describe": None,
    # density_df: p_total / p_sev / S / F / lev / loss + any p_/exeqa_ regex
    # the unit happens to have. We keep an explicit set of columns and let
    # the capture script intersect with what's available.
    "density_df": ["loss", "p_total", "p_sev", "F", "S", "lev"],
}

PORT_FRAMES = {
    "stats_df": None,
    "describe": None,
    # density_df keeps the per-line p_/exeqa_ family plus the totals.
    # The capture script expands the regex against the actual columns.
    "density_df": {
        "regex": r"^(loss|S|F|p_total|p_[^t]|exeqa_)",
    },
}

# Augmented-df columns per distortion (snapshot only the load-bearing ones).
AUGMENTED_COLUMNS = {
    "regex": r"^(loss|S|gS|gp_total|exag_)",
}
