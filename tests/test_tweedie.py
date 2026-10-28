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

And the bare ``tweedie`` statement with its ``TweedieConversion`` answer, plus
the ``cv`` form of the clause (``[Tweedie-Statement]``,
``dev/done/plan-tweedie-statement.md``).
"""
from __future__ import annotations

import numpy as np
import pytest

from aggregate import build, tweedie_convert, tweedie_density
from aggregate.tweedie import Tweedie, TweedieConversion, TweedieParameters


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


# ----------------------------------------------------------------------
# The bare statement: [Tweedie-Statement]
# ----------------------------------------------------------------------

def _fields(obj):
    """The eight fields of a ``TweedieConversion`` as a float array."""
    return np.array(obj, dtype=float)


def test_tweedie_statement_worked_example():
    """``tweedie 1.5 100 2`` is ten claims of a gamma with mean 10 and CV 1."""
    t = build('tweedie 1.5 100 2')
    assert isinstance(t, TweedieConversion)
    assert np.isclose(t.frequency, 10, rtol=1e-12)
    assert np.isclose(t.sev_m, 10, rtol=1e-12)
    assert np.isclose(t.sev_cv, 1, rtol=1e-12)
    assert np.isclose(t.cv, 0.447213595499958, rtol=1e-12)
    assert np.isclose(t.p0, np.exp(-10), rtol=1e-12)
    # the compound variance is the Tweedie variance
    assert np.isclose(t.frequency * t.sev_m ** 2 * (1 + t.sev_cv ** 2),
                      t.dispersion * t.mean ** t.p, rtol=1e-12)


@pytest.mark.parametrize('p', [1.005, 1.5, 1.995])
def test_tweedie_statement_three_forms_agree(p):
    """The dispersion, CV and compound forms of one law give the same fields."""
    t = build(f'tweedie {p} 100 2')
    by_cv = build(f'tweedie {p} 100 cv {t.cv!r}')
    by_compound = build(f'tweedie {t.frequency!r} claims sev gamma '
                        f'{t.sev_m!r} cv {t.sev_cv!r}')
    assert np.allclose(_fields(by_cv), _fields(t), rtol=1e-12)
    assert np.allclose(_fields(by_compound), _fields(t), rtol=1e-12)


@pytest.mark.parametrize('sev', ['gamma 10 cv 1', '10 * gamma 1',
                                 '0.5 * gamma 20 cv 1'])
def test_tweedie_statement_every_gamma_spelling(sev):
    """Any spelling of the same gamma an ``agg`` accepts gives the same answer."""
    t = build(f'tweedie 10 claims sev {sev}')
    assert np.allclose(_fields(t), _fields(build('tweedie 1.5 100 2')),
                       rtol=1e-12)


@pytest.mark.parametrize('program', [
    'tweedie 10 claims sev lognorm 10 cv 1',
    'tweedie 10 claims sev gamma 10 cv 1 + 5',
    'tweedie 10 claims sev gamma [10 20] cv 1 wts [.5 .5]',
    'tweedie 10 claims sev [gamma lognorm] [10 20] cv 1 wts [.5 .5]',
    'tweedie 10 claims sev gamma 10 cv 1 !',
    'tweedie 0 claims sev gamma 10 cv 1',
    'tweedie 10 claims sev gamma 0',
    'tweedie 1 100 2',
    'tweedie 2 100 2',
    'tweedie 1 100 cv 0.5',
    'tweedie 2.5 100 cv 0.5',
    'tweedie 1.5 0 2',
    'tweedie 1.5 100 0',
    'tweedie 1.5 100 cv 0',
])
def test_tweedie_statement_refusals_name_the_form(program):
    """Every refusal is a ValueError naming the statement, never a ZeroDivisionError."""
    with pytest.raises(ValueError, match='tweedie'):
        build(program)


def test_tweedie_statement_string_is_ascii_and_pastes_back():
    """Each spelling in the answer is legal input for the same law."""
    t = build('tweedie 1.5 100 2')
    text = str(t)
    text.encode('ascii')
    assert text == ('tweedie 1.5 100 2 = tweedie 1.5 100 cv 0.447214 = '
                    'tweedie 10 claims sev gamma 10 cv 1, Pr(X=0) 4.53999e-05')
    spellings = text.split(', Pr(X=0)')[0].split(' = ')
    assert len(spellings) == 3
    for s in spellings:
        assert np.allclose(_fields(build(s)), _fields(t), rtol=1e-5)


def test_tweedie_statement_stores_nothing():
    """A conversion is an answer: the recipe base does not change."""
    before = set(build._recipes)
    rv = build.build_many('tweedie 1.5 100 2')
    assert set(build._recipes) == before
    assert len(rv) == 1
    assert rv[0].kind == 'conversion'
    assert isinstance(rv[0].object, TweedieConversion)


def test_bare_expression_unchanged():
    """The other answer kind still evaluates to its number."""
    assert build('(2+2)') == 4


def test_tweedie_statement_agrees_with_recognition():
    """The calculator and ``as_tweedie`` agree on a long-hand compound."""
    a = build('agg TWREC 10 claims sev gamma 10 cv 1 poisson')
    t = build('tweedie 10 claims sev gamma 10 cv 1')
    assert np.allclose(a.as_tweedie(), (t.p, t.mean, t.dispersion), rtol=1e-12)
    assert TweedieConversion.from_dispersion(*a.as_tweedie()) == t


# ----------------------------------------------------------------------
# The CV form inside an agg: [Tweedie-Clause-CV]
# ----------------------------------------------------------------------

def test_tweedie_clause_cv_form_round_trips():
    """``tweedie <p> <mean> cv <cv>`` builds and renders as declared."""
    a = build('agg TWCV tweedie 1.5 100 cv 0.447213595499958')
    assert 'tweedie 1.5 100 cv 0.447213595499958' in a.pprogram
    assert np.isclose(a.as_tweedie().dispersion, 2, rtol=1e-12)
    b = build(a.pprogram)
    assert b.pprogram == a.pprogram
    assert b.spec == a.spec


def test_tweedie_clause_cv_form_inside_a_pnl():
    """The clause keeps its CV form as the engine of a P&L."""
    p = build('pnl TWPNL 1100 premium less agg TWE tweedie 1.5 100 cv 0.5')
    assert 'tweedie 1.5 100 cv 0.5' in p.pprogram


def test_tweedie_cv_provenance_leaves_hashes_alone():
    """``_tweedie_cv`` is dropped when unset, so no existing hash moved.

    The two hashes were measured at 1.21.0, before the key existed. A
    deliberate change to the constructor that moves every hash re-pins them.
    """
    a = build('agg TWH1 tweedie 1.5 100 2', update=False)
    b = build('agg TWH2 10 claims sev lognorm 50 cv 1 poisson', update=False)
    assert '_tweedie_cv' not in a.spec and '_tweedie_cv' not in b.spec
    assert a._spec_hash() == '618DBE6E'
    assert b._spec_hash() == 'F0160E2C'
