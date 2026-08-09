"""Regression tests for the Tweedie module.

Lightweight in-regression coverage of the three public surfaces in
``aggregate.tweedie``: the parameter-translator ``tweedie_convert``,
the series-expansion density ``tweedie_density``, and the ``Tweedie``
class. Restricted to the compound-Poisson-gamma regime (1 < p < 2)
where closed-form moments are available from V(μ) = dispersion · μ^p.
Heavy-tail regimes (p > 2, p < 0, Cauchy at p=∞) need Fourier
inversion and tolerance-tuning that doesn't belong in a fast suite.

Also covers the ``tweedie`` DecL clause surviving the parse
(``[Tweedie-Round-Trip]``, 1.0.0a231, ``dev/plan-tweedie.md``): the clause
renders back instead of its compound-Poisson-gamma expansion, the author's
``note{}`` is no longer overwritten, and the argument order is
``<p> <mean> <dispersion>``.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build, tweedie_convert, tweedie_density
from aggregate.tweedie import Tweedie, TweedieParameters


# (p, mean, dispersion) — compound-Poisson-gamma regime.
PCASES = [
    (1.5, 10.0, 1.0),
    (1.5, 100.0, 0.5),
    (1.1, 10.0, 1.0),
    (1.9, 10.0, 1.0),
]


@pytest.mark.parametrize("p,mu,disp", PCASES)
def test_tweedie_convert_roundtrip(p, mu, disp):
    """(p, μ, σ²) → (λ, α, β) → (μ, σ²) is the identity."""
    fwd = tweedie_convert(p=p, μ=mu, σ2=disp)
    back = tweedie_convert(λ=fwd["λ"], α=fwd["α"], β=fwd["β"])
    assert np.isclose(back["μ"], mu, rtol=1e-10)
    assert np.isclose(back["σ^2"], disp, rtol=1e-10)
    assert np.isclose(back["p"], p, rtol=1e-10)


@pytest.mark.parametrize("p,mu,disp", PCASES)
def test_tweedie_density_finite_positive(p, mu, disp):
    """Density at x = μ is finite and strictly positive."""
    d = tweedie_density(mu, p=p, μ=mu, σ2=disp)
    assert np.isfinite(d)
    assert d > 0


@pytest.mark.parametrize("p,mu,disp", PCASES)
def test_tweedie_moments_match_definition(p, mu, disp):
    """Tweedie reproductive moments satisfy mean=μ, var = disp · μ^p."""
    tw = Tweedie(p, mean=mu, dispersion=disp)
    m, v, _ = tw.stats()
    assert np.isclose(m, mu, rtol=1e-4)
    assert np.isclose(v, disp * mu**p, rtol=1e-4)


def test_tweedie_dual_involution():
    """``Tweedie(...).dual().dual()`` returns the same reproductive params."""
    tw = Tweedie(1.5, mean=100.0, dispersion=0.5)
    back = tw.dual().dual()
    assert np.isclose(back.mean, tw.mean, rtol=1e-12)
    assert np.isclose(back.dispersion, tw.dispersion, rtol=1e-12)


def test_tweedie_reproductive_triple_is_positional():
    """``Tweedie(*params)`` splats, which is why the keyword marker came out."""
    params = TweedieParameters(p=1.5, mean=100.0, dispersion=0.5)
    tw = Tweedie(*params)
    assert (tw.p, tw.mean, tw.dispersion) == (1.5, 100.0, 0.5)


# ----------------------------------------------------------------------
# The DecL clause: [Tweedie-Round-Trip]
# ----------------------------------------------------------------------

def test_tweedie_clause_round_trips():
    """The clause renders back, not its compound-Poisson-gamma expansion."""
    a = build('agg TWRT tweedie 1.005 1 0.1')
    assert 'tweedie 1.005 1 0.1' in a.pprogram
    assert 'poisson' not in a.pprogram
    assert a._tweedie == TweedieParameters(p=1.005, mean=1.0, dispersion=0.1)


def test_tweedie_clause_note_survives():
    """The author's ``note{}`` is not overwritten by a machine string.

    The clobber it guards against was live from the SLY parser through
    1.0.0a230: every ``tweedie`` line in ``library.agg`` lost its authored note
    to ``Tw(p=...) --> CP(...)``.
    """
    a = build('agg TWNOTE tweedie 1.005 1 0.1 note{mine} tags{x}')
    assert a.note == 'mine'
    assert a.tags == ('x',)
    assert 'note{mine}' in a.format_program(layout='terse', trailer=True)


def test_tweedie_clause_engine_is_the_expansion():
    """Provenance changes the text, never the distribution."""
    a = build('agg TWENG tweedie 1.005 1 0.1')
    assert a.spec['freq_name'] == 'poisson'
    assert a.spec['sev_name'] == 'gamma'
    # mean = μ and variance = dispersion * μ ** p, by definition.
    assert np.isclose(a.est_m, 1.0, rtol=1e-6)
    assert np.isclose(a.est_sd ** 2, 0.1 * 1.0 ** 1.005, rtol=1e-4)


@pytest.mark.parametrize('p', [1, 2, 10, 0.5, -1])
def test_tweedie_clause_rejects_p_outside_the_open_unit_interval(p):
    """``p`` outside (1, 2) is a named error, not a ZeroDivisionError.

    Doubles as the guard on the 1.0.0a231 argument-order change: an old
    ``<mean> <p> <dispersion>`` program reads its mean as ``p`` and almost
    always lands outside the interval.
    """
    with pytest.raises(Exception, match='(?s)tweedie.*p must be strictly'):
        build(f'agg TWBAD tweedie {p} 1.005 0.1')


def test_tweedie_to_decl_round_trips_through_build():
    """``Tweedie`` to DecL to ``Aggregate`` and back to the same parameters."""
    tw = Tweedie(1.5, mean=100.0, dispersion=0.5)
    a = build(tw.to_decl('TWDECL'))
    assert a._tweedie == TweedieParameters(p=1.5, mean=100.0, dispersion=0.5)


def test_tweedie_note_is_ascii_renderable():
    """No Greek in a user-facing string; the old note crashed a cp1252 console."""
    a = build('agg TWASCII tweedie 1.005 1 0.1')
    for text in (a.note, a.program, a.pprogram, repr(a)):
        text.encode('cp1252')


def test_non_tweedie_spec_carries_no_provenance():
    """``_tweedie`` is dropped when unset, so ``_spec_hash`` did not move."""
    a = build('agg TWPLAIN 10 claims sev lognorm 50 cv 1 poisson')
    assert '_tweedie' not in a.spec
    assert a._tweedie is None
