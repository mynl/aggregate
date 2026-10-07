"""``spec_to_decl`` rejects a dense constructor-argument spec instead of lying.

`[Unparse-Dense-Spec-Guard]` (``dev/plan-a400-v1-loose-ends.md`` phase 2).
``spec_to_decl`` documents that it takes the **sparse** parser spec, the third
element of ``Underwriter.parser.parse(...)``. Handed the **dense**
``Aggregate.spec`` instead it used to emit DecL that was quietly wrong rather
than obviously broken, because the dense dict spells "unset" as ``0`` or
``None`` where the parser omits the key, and ``0`` is legitimate for
``exp_premium`` and ``sev_scale``:

* ``13.7376 claims`` rendered as ``0 premium at 0 lr``,
* the severity picked up a ``0 *`` scale,
* a spurious ``poisson 0 0 loss`` appeared.

That text re-parsed and *built*, to ``est_m = 0`` and ``est_cv = nan``, with no
error anywhere. The only reason it was not already biting is that it first
raised ``AttributeError`` on ``label_map``, a present-but-``None`` value
defeating a ``.get('label_map', {})`` default. Both are fixed here: the
``label_map`` reads are ``None``-tolerant, so the guard is what fires.

This is a guard, deliberately not a decompiler. Turning a built object back
into DecL needs a per-key inverse of the constructor's defaulting, which is a
separate and much larger question.
"""

import inspect
import warnings

import pytest

from aggregate import build, Underwriter
from aggregate.decl_writer import spec_to_decl, _is_dense_spec, _constructor_params
from aggregate.distributions import Aggregate

PROGRAMS = [
    'agg DG.Plain 13.7376 claims sev lognorm 100 cv 2 poisson',
    'agg DG.Premium 1000 premium at 0.65 lr sev lognorm 50 cv 1.5 poisson',
    'agg DG.Layered 10 claims 500 xs 100 sev lognorm 100 cv 2 poisson',
    'agg DG.Reins 10 claims sev lognorm 100 cv 2 occurrence net of 50 xs 50 poisson',
    'agg DG.Dice dfreq [3] dsev [1:6]',
]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


@pytest.fixture(scope='module')
def uw():
    return Underwriter()


def _sparse(uw, program):
    """The parser's (kind, name, spec) triple: the documented input shape."""
    return uw.parser.parse(uw.lexer.tokenize(program))


# ---------------------------------------------------------------------------
# The guard fires on the dense shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program', PROGRAMS)
def test_dense_spec_raises_value_error(program):
    ob = build(program)
    with pytest.raises(ValueError, match='dense constructor-argument spec'):
        spec_to_decl(ob.spec)


@pytest.mark.parametrize('program', PROGRAMS)
def test_dense_spec_is_detected(program):
    assert _is_dense_spec(build(program).spec, 'agg')


def test_the_error_names_the_fix():
    """A guard that does not say what to do instead is half a guard."""
    ob = build(PROGRAMS[0])
    with pytest.raises(ValueError) as ei:
        spec_to_decl(ob.spec)
    msg = str(ei.value)
    assert 'sparse parser spec' in msg
    assert '.program' in msg          # the route that actually works
    assert 'to_agg' in msg


def test_dense_spec_no_longer_dies_on_label_map():
    """The old failure was an incidental AttributeError that masked the real one.

    ``label_map`` is present-but-``None`` on a dense spec, which defeated the
    ``.get('label_map', {})`` default. The guard must be what the caller sees.
    """
    ob = build(PROGRAMS[0])
    assert ob.spec.get('label_map') is None, 'premise of this test'
    with pytest.raises(ValueError):          # not AttributeError
        spec_to_decl(ob.spec)


# ---------------------------------------------------------------------------
# The guard does not fire on the documented input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('program', PROGRAMS)
def test_sparse_spec_still_renders(uw, program):
    kind, name, spec = _sparse(uw, program)
    assert not _is_dense_spec(spec, kind)
    out = spec_to_decl(spec, kind, name)
    assert out.startswith(f'{kind} {name}')


@pytest.mark.parametrize('program', PROGRAMS)
def test_sparse_spec_round_trips_through_the_guard(uw, program):
    """Render, re-parse, render again: the canonical form is a fixed point."""
    kind, name, spec = _sparse(uw, program)
    once = spec_to_decl(spec, kind, name)
    k2, n2, s2 = _sparse(uw, once)
    assert spec_to_decl(s2, k2, n2) == once


# ---------------------------------------------------------------------------
# The discriminator itself
# ---------------------------------------------------------------------------

def test_discriminator_excludes_private_constructor_params():
    """``Aggregate.spec`` omits ``_tweedie``, so private params must not count.

    Including them made the subset test miss by exactly one key, which is how
    the first cut of this guard silently failed to fire.
    """
    params = _constructor_params('agg')
    assert params, 'params must resolve'
    assert not any(n.startswith('_') for n in params)
    assert '_tweedie' in inspect.signature(Aggregate.__init__).parameters
    assert '_tweedie' not in params


def test_discriminator_is_quiet_on_an_empty_or_tiny_spec():
    assert not _is_dense_spec({}, 'agg')
    assert not _is_dense_spec({'name': 'x', 'exp_en': 10}, 'agg')


def test_discriminator_is_quiet_on_kinds_with_no_constructor():
    """port / bvagg / distortion have no dense shape here, so never flag."""
    for kind in ('port', 'bvagg', 'distortion', 'pnl'):
        assert _constructor_params(kind) == frozenset()
        assert not _is_dense_spec({'anything': 1}, kind)


def test_margin_between_the_two_shapes_is_wide(uw):
    """The whole corpus must stay far from tripping the guard.

    Measured over the shipped ``.agg`` corpus the richest parser spec carries
    far fewer keys than the constructor has parameters. If that margin ever
    narrows to nothing the guard needs a different discriminator, so pin it.
    """
    from pathlib import Path
    from aggregate.parser import UnderwritingLexer
    widest, flagged, n = 0, 0, 0
    for f in sorted(Path('src/aggregate/agg').glob('*.agg')):
        for line in UnderwritingLexer.preprocess(f.read_text(encoding='utf-8')):
            try:
                kind, name, spec = _sparse(uw, line)
            except Exception:
                continue
            n += 1
            widest = max(widest, len(spec))
            flagged += _is_dense_spec(spec, kind)
    assert n > 500, 'corpus should be substantial'
    assert flagged == 0, 'a parser spec was mistaken for a dense spec'
    assert widest < len(_constructor_params('agg')), 'margin has closed'
