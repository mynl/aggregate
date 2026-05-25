"""Parametric tests for splice / splice-and-layer cases from test_suite2.agg.

Twelve cases hand-curated by the user that exercise the `sev_lb` / `sev_ub`
splice path with and without policy layers. Four of them describe
measure-zero splices (the splice window does not intersect the underlying
distribution's support) and must raise ``ValueError`` at construction time.

Lives alongside (not inside) ``test_suite.agg`` so the established 134-line
parametric test count doesn't move; the two files will be consolidated in
a later pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aggregate.distributions import Aggregate, Severity
from aggregate.parser import UnderwritingLexer
from aggregate.underwriter import Underwriter

SUITE_PATH = Path(__file__).parent.parent / "src" / "aggregate" / "agg" / "test_suite2.agg"

# Names of the four cases that describe measure-zero splices — the splice
# window lies entirely outside the underlying distribution's support, so the
# conditional distribution is mathematically undefined.
INVALID_NAMES = {"Splice02", "Splice05", "Splice07", "Splice10"}


def _load_lines() -> list[str]:
    text = SUITE_PATH.read_text(encoding="utf-8")
    return UnderwritingLexer.preprocess(text)


def _line_id(line: str) -> str:
    return (line[:80] + "...") if len(line) > 80 else line


LINES = _load_lines()


@pytest.fixture(scope="module")
def underwriter() -> Underwriter:
    """A fresh Underwriter for the splice suite — no databases preloaded."""
    return Underwriter()


def _short_name(name: str) -> str:
    """Strip the ``G.`` prefix to match INVALID_NAMES."""
    return name.split(".", 1)[1] if "." in name else name


@pytest.mark.parametrize("line", LINES, ids=[_line_id(ln) for ln in LINES])
def test_splice_line_parses(line: str, underwriter):
    """All 12 lines must parse — measure-zero is a construction-time error.

    Parsing the spec must succeed even for the four invalid cases; the
    ``ValueError`` only fires when the Severity is actually constructed.
    """
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    assert kind in {"sev", "agg"}, f"Unexpected kind {kind!r} for line: {line}"
    assert isinstance(name, str) and name, f"Empty name for line: {line}"
    assert isinstance(spec, dict), f"Spec should be a dict for kind {kind}"


@pytest.mark.parametrize("line", LINES, ids=[_line_id(ln) for ln in LINES])
def test_splice_line_builds_or_raises(line: str, underwriter):
    """Valid splice cases build; measure-zero ones raise ``ValueError``.

    Builds the spec by routing through ``Underwriter._factory`` (the same
    code path as ``build('agg ...')`` / ``build('sev ...')``); this exercises
    the Severity constructor under the same conditions a user-typed program
    would.
    """
    kind, name, spec = underwriter.parser.parse(underwriter.lexer.tokenize(line))
    short = _short_name(name)

    if short in INVALID_NAMES:
        with pytest.raises(ValueError, match="zero probability mass"):
            _build_from_spec(kind, spec)
        return

    obj = _build_from_spec(kind, spec)
    if kind == "sev":
        assert isinstance(obj, Severity), f"{name}: expected Severity, got {type(obj).__name__}"
    elif kind == "agg":
        assert isinstance(obj, Aggregate), f"{name}: expected Aggregate, got {type(obj).__name__}"


def _build_from_spec(kind: str, spec: dict):
    """Instantiate the appropriate top-level object from a parsed spec.

    Mirrors the dispatch in ``Underwriter._factory`` for the two kinds that
    appear in ``test_suite2.agg``. We don't go through ``Underwriter.write``
    because the splice failures need to surface as a ``ValueError`` at the
    Severity layer; the Underwriter wraps errors and rebuilds.
    """
    if kind == "sev":
        return Severity(**spec)
    if kind == "agg":
        return Aggregate(**spec)
    raise ValueError(f"Unhandled kind {kind!r}")


# ---- Moments / stats on a spliced severity ----------------------------------
# Regression for the bug where ``s.stats()`` on a spliced severity blew up with
# a TypeError because the DecL parser passes ``sev_lb`` / ``sev_ub`` as
# length-1 lists. ``np.where`` in ``make_conditional_*`` then returned a
# ``(1,)``-shaped array that ``scipy.integrate.quad`` could not consume.
# The fix coerces the bounds to scalars in ``Severity.__init__``.


def test_spliced_severity_stats_does_not_crash():
    """``s.stats()`` on a spliced severity returns finite mean and variance."""
    from aggregate import build

    s = build(" sev X lognorm 10 cv .5 splice [20 30] ")
    mean, var = s.stats()
    assert 20 <= float(mean) <= 30, f"mean {mean} should lie inside the splice [20, 30]"
    assert float(var) > 0, f"variance {var} should be positive"


def test_spliced_severity_moms_returns_three_finite_moments():
    """``s.moms()`` returns three positive, monotone-increasing raw moments."""
    from aggregate import build

    s = build(" sev X lognorm 10 cv .5 splice [20 30] ")
    m1, m2, m3 = s.moms()
    assert 20 <= m1 <= 30
    # Raw moments of a positive variable: E[X^k] is strictly increasing in k.
    assert m1 ** 2 <= m2
    assert m1 * m2 <= m3


def test_scalar_bound_rejects_multi_element():
    """``_scalar_bound`` raises a clear error on a (mythical) multi-segment splice."""
    from aggregate.distributions import _scalar_bound

    with pytest.raises(ValueError, match="Multi-segment splice"):
        _scalar_bound([1.0, 2.0])


# ---- Layered moments on a spliced severity (Option A regressions) -----------
# Pins the analytical answers for the two G.Splice11/G.Splice12 shapes — these
# silently returned (0, 0, 0) or NaN before Option A separated the layer
# wrappers from ``self.fz.<method>``. Both verify the splice + layer composition
# under conditional severity moments.


def _spliced_uniform_with_policy(attachment: float, limit: float):
    """Build a 1-claim Aggregate spec mirroring G.Splice11/G.Splice12 shapes."""
    from aggregate import build

    program = (
        f' agg LayerSpliceTest 1 claim {limit} xs {attachment} '
        f'sev 10 * uniform + 5 splice [8 12] fixed '
    )
    return build(program, update=True).sevs[0]


def test_g_splice11_moments_full_limit():
    """Spliced uniform [8,12] with policy 5 xs 3: all claims are full limit = 5.

    Analytically E[claim^k] = 5^k for k = 1, 2, 3.
    """
    sev = _spliced_uniform_with_policy(attachment=3, limit=5)
    m1, m2, m3 = sev.moms()
    assert m1 == pytest.approx(5.0, abs=1e-9), f'expected 5, got {m1}'
    assert m2 == pytest.approx(25.0, abs=1e-9), f'expected 25, got {m2}'
    assert m3 == pytest.approx(125.0, abs=1e-9), f'expected 125, got {m3}'


def test_g_splice12_moments_excess_layer():
    """Spliced uniform [8,12] with policy 2 xs 8: proper excess layer.

    For X ∈ [8, 10]: claim = X - 8 ~ Uniform[0, 2].
    For X ∈ [10, 12]: claim = 2 (full limit).
    Each half has probability 0.5 under the spliced uniform.

    Analytical moments:
      E[claim]   = 0.5 * 1   + 0.5 * 2  = 1.5
      E[claim²]  = 0.5 * 4/3 + 0.5 * 4  = 8/3
      E[claim³]  = 0.5 * 2   + 0.5 * 8  = 5
    """
    sev = _spliced_uniform_with_policy(attachment=8, limit=2)
    m1, m2, m3 = sev.moms()
    assert m1 == pytest.approx(1.5, abs=1e-6), f'expected 1.5, got {m1}'
    assert m2 == pytest.approx(8 / 3, abs=1e-6), f'expected 8/3, got {m2}'
    assert m3 == pytest.approx(5.0, abs=1e-6), f'expected 5, got {m3}'


def test_layer_wrappers_separate_from_fz_methods():
    """Option A invariant: ``self.fz.<method>`` stays splice-only; layered
    wrappers live on ``self._layered_<method>``.

    A spliced severity's ``fz.sf(x)`` at an in-splice point should return a
    value in (0, 1) (the splice-conditional survival), not 0 (the layered
    survival above limit).
    """
    from aggregate import build

    s = build(' sev X 10 * uniform + 5 splice [8 12] ')
    # Spliced support is [8, 12]; sf(9) under splice = (12-9)/(12-8) = 0.75.
    underlying_sf_at_9 = float(s.fz.sf(9))
    assert 0.7 < underlying_sf_at_9 < 0.8, (
        f'fz.sf(9) should be the splice-only survival ~0.75; got {underlying_sf_at_9}. '
        f'If close to 0, the layer transform leaked onto self.fz (Option A regression).'
    )
    # And ``_layered_sf`` exists and produces a different answer when a layer
    # is applied — Sanity-check the attribute is present.
    assert hasattr(s, '_layered_sf')
