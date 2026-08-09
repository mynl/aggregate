"""Regression tests for the Tweedie module.

Lightweight in-regression coverage of the three public surfaces in
``aggregate.tweedie``: the parameter-translator ``tweedie_convert``,
the series-expansion density ``tweedie_density``, and the ``Tweedie``
class. Restricted to the compound-Poisson-gamma regime (1 < p < 2)
where closed-form moments are available from V(μ) = dispersion · μ^p.
Heavy-tail regimes (p > 2, p < 0, Cauchy at p=∞) need Fourier
inversion and tolerance-tuning that doesn't belong in a fast suite.

Also covers the ``tweedie`` DecL clause surviving the parse
(``[Tweedie-Round-Trip]``, 1.0.0a231, ``dev/done/plan-tweedie.md``): the clause
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


# ----------------------------------------------------------------------
# Aggregate.as_tweedie: [Tweedie-Live-Object]
# ----------------------------------------------------------------------

#: The same distribution three ways: the keyword, and the two long-hand
#: spellings shipped as ``TweedieDirect`` / ``TweedieFromMoments``.
_SAME_TWEEDIE = [
    'agg TW3A tweedie 1.005 1 0.1',
    'agg TW3B 10.050251256281404 claims '
    'sev gamma 0.0995 cv 0.07088812050083283 poisson',
    'agg TW3C 10.050251256281404 claims sev 0.0005 * gamma 199 poisson',
]


@pytest.mark.parametrize('program', _SAME_TWEEDIE)
def test_as_tweedie_recognizes_every_spelling(program):
    """Declared or long-hand, a compound Poisson-gamma reports its parameters.

    Recognition, not provenance: only the first of these carries ``_tweedie``.
    """
    t = build(program).as_tweedie()
    assert t is not None
    assert np.isclose(t.p, 1.005, rtol=1e-12)
    assert np.isclose(t.mean, 1.0, rtol=1e-12)
    assert np.isclose(t.dispersion, 0.1, rtol=1e-12)


def test_as_tweedie_matches_the_variance_function():
    """The reported triple satisfies var = dispersion * mean ** p."""
    a = build('agg TWVAR 10 claims sev gamma 5 cv 0.5 poisson')
    t = a.as_tweedie()
    assert np.isclose(a.est_m, t.mean, rtol=1e-6)
    assert np.isclose(a.est_sd ** 2, t.dispersion * t.mean ** t.p, rtol=1e-6)


def test_as_tweedie_declared_wins_over_derivation():
    """A declared triple comes back verbatim, not round-tripped through floats."""
    a = build('agg TWEXACT tweedie 1.005 1 0.1')
    assert a.as_tweedie() == TweedieParameters(p=1.005, mean=1.0,
                                               dispersion=0.1)


def test_as_tweedie_splats_into_the_class():
    """The whole point of the field order and the positional constructor."""
    a = build('agg TWSPLAT tweedie 1.5 100 0.5')
    tw = Tweedie(*a.as_tweedie())
    assert np.isclose(tw.mean, 100.0)
    assert np.isclose(tw.dispersion, 0.5)
    assert np.isclose(tw.p, 1.5)


@pytest.mark.parametrize('program', [
    'agg TWN1 10 claims sev lognorm 5 cv 0.5 poisson',            # not gamma
    'agg TWN2 10 claims sev gamma 5 cv 0.5 mixed gamma 0.3',      # not poisson
    'agg TWN3 10 claims sev gamma 5 cv 0.5 poisson zt',           # zero-modified
    'agg TWN4 10 claims 4 xs 0 sev gamma 5 cv 0.5 poisson',       # limited
    'agg TWN5 10 claims sev gamma 5 cv 0.5 + 2 poisson',          # shifted
    'agg TWN6 10 claims sev gamma [5 8] cv 0.5 wts [.5 .5] poisson',   # mixed
    'agg TWN7 [5 5] claims [10 20] xs 0 sev gamma 5 cv 0.5 poisson',   # profile
    'agg TWN8 10 claims sev gamma 5 cv 0.5 occurrence net of 2 xs 3 poisson',
    'agg TWN9 10 claims sev gamma 5 cv 0.5 poisson aggregate net of 20 xs 30',
    'agg TWNA 10 claims sev gamma 5 cv 0.5 poisson approximate slognorm',
])
def test_as_tweedie_refuses_anything_not_a_bare_compound_poisson_gamma(program):
    """Every gate, one case each. None of these has reproductive parameters."""
    assert build(program).as_tweedie() is None


def test_recognition_does_not_leak_into_the_unparser():
    """A long-hand compound Poisson-gamma still renders the way it was written.

    The provenance / recognition split: ``as_tweedie`` is generous because it
    costs nothing, the writer is strict because rewriting an author's program
    into a spelling they did not choose is not its job.
    """
    a = build('agg TWLONG 10.050251256281404 claims '
              'sev gamma 0.0995 cv 0.07088812050083283 poisson')
    assert a.as_tweedie() is not None
    assert 'tweedie' not in a.pprogram
    assert 'poisson' in a.pprogram
