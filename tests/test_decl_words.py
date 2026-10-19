"""The DecL word list, :func:`aggregate.decl_words.decl_words`.

Each group is checked against what decides it: the grammar for the groups read
from ``decl.lark``, scipy and the library's special cases for the severity
names, the frequency registry for the mixing distributions, and the distortion
registry by construction. Then the JSON entry point, and the lexer built from
the list. ``dev/plan-decl-word-list.md`` (``[DecL-Word-List]``).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import scipy.stats as ss
from pygments.token import Name

import aggregate
from aggregate import Distortion
from aggregate._frequency import Frequency
from aggregate.decl_pygments import AggLexer
from aggregate.decl_words import _grammar_reserved_words, decl_words

WORDS = decl_words()

#: The severity names the library resolves itself rather than through
#: ``getattr(scipy.stats, name)`` (``_severity._classify_sev``).
SPECIAL_SEVERITIES = {'cantor', 'dhistogram', 'chistogram'}

#: The frequency registry entries that are not mixing distributions: the
#: empirical and renewal frequencies have their own declarations.
NOT_MIXING = {'empirical', 'renewal'}

#: One valid parameter set per documented distortion kind. A kind added to the
#: registry fails :func:`test_every_distortion_kind_has_an_example` until it
#: gets one here.
DISTORTION_EXAMPLES = {
    'beta': dict(a=0.5, b=1.5),
    'bitvar': dict(p0=0.95, p1=0.99, w1=0.5),
    'ccoc': dict(r=0.1),
    'clin': dict(slope=1.2),
    'cll': dict(b=0.5),
    'dual': dict(shape=1.5),
    'lep': dict(r=0.1),
    'ly': dict(r=0.1),
    'ph': dict(shape=0.5),
    'tvar': dict(shape=0.2),
    'wang': dict(shape=0.3),
    'wtdtvar': dict(ps=[0.5, 0.9], wts=[0.3, 0.7]),
}

#: The lexer's token type for a severity name, by shape parameter count.
SEVERITY_TOKENS = {0: Name.Function, 1: Name.Class, 2: Name.Namespace,
                   None: Name.Class}


def _first_token(source: str):
    """The type of the first significant token AggLexer emits for ``source``."""
    return next(t for t, v in AggLexer().get_tokens(source) if v.strip())


def test_groups_and_their_order():
    assert list(WORDS) == ['reserved', 'frequencies', 'declarations', 'clauses',
                           'constants', 'severities', 'mixing', 'distortions']


@pytest.mark.parametrize('group', [g for g in WORDS if g != 'severities'])
def test_group_is_a_sorted_tuple_of_distinct_words(group):
    words = WORDS[group]
    assert isinstance(words, tuple)
    assert list(words) == sorted(set(words))


def test_severities_sorted_by_name():
    assert list(WORDS['severities']) == sorted(WORDS['severities'])


def test_reserved_is_the_grammar_id_exclusion_list():
    assert WORDS['reserved'] == _grammar_reserved_words()
    for word in ('agg', 'poisson', 'bivariate', 'dbvsev', 'pnl', 'zt', 'mixed'):
        assert word in WORDS['reserved']


def test_frequencies_and_declarations_are_reserved():
    reserved = set(WORDS['reserved'])
    assert set(WORDS['frequencies']) <= reserved
    assert set(WORDS['declarations']) <= reserved
    assert 'poisson' in WORDS['frequencies']


def test_clauses_and_constants_are_not_reserved():
    """A clause terminal carries its brace and ``inf`` is a number, so the bare
    words are legal identifiers and absent from the reserved list."""
    assert WORDS['clauses'] == ('hints', 'note', 'tags')
    assert WORDS['constants'] == ('inf',)
    assert not set(WORDS['clauses'] + WORDS['constants']) & set(WORDS['reserved'])


@pytest.mark.parametrize('name', sorted(WORDS['severities']))
def test_severity_name_resolves_with_its_shape_count(name):
    count = WORDS['severities'][name]
    if name in SPECIAL_SEVERITIES:
        assert count == (1 if name == 'cantor' else None)
        return
    dist = getattr(ss, name)
    assert dist.numargs == count


def test_severities_are_not_reserved():
    assert not set(WORDS['severities']) & set(WORDS['reserved'])


def test_mixing_is_the_frequency_registry_less_the_named_frequencies():
    expected = set(Frequency._registry) - set(WORDS['frequencies']) - NOT_MIXING
    assert set(WORDS['mixing']) == expected


def test_every_distortion_kind_has_an_example():
    assert set(WORDS['distortions']) == set(DISTORTION_EXAMPLES)
    assert WORDS['distortions'] == tuple(sorted(
        Distortion.available_distortions(pricing=False)))


@pytest.mark.parametrize('kind', WORDS['distortions'])
def test_distortion_name_builds_its_kind(kind):
    assert Distortion(kind, **DISTORTION_EXAMPLES[kind]).kind == kind


def test_beta_is_the_one_distortion_name_in_another_group():
    """The consumer rule in the module docstring exists for this overlap. A new
    one is not wrong, but it is worth knowing about."""
    others = set().union(*(set(w) for g, w in WORDS.items() if g != 'distortions'))
    assert set(WORDS['distortions']) & others == {'beta'}


def test_each_call_returns_a_fresh_copy():
    first = decl_words()
    first['severities']['made_up'] = 9
    first['reserved'] = ()
    again = decl_words()
    assert 'made_up' not in again['severities']
    assert again['reserved'] == WORDS['reserved']


def test_module_entry_point_prints_the_list_as_json():
    """``python -m aggregate.decl_words``, in a subprocess so the run is the one
    a web build sees. Nothing on stderr: in particular no ``runpy`` warning."""
    result = subprocess.run([sys.executable, '-m', 'aggregate.decl_words'],
                            capture_output=True, text=True, check=True)
    assert result.stderr == ''
    payload = json.loads(result.stdout)
    assert payload['version'] == aggregate.__version__
    assert payload['groups'] == json.loads(json.dumps(WORDS))


@pytest.mark.parametrize('name', sorted(WORDS['severities']))
def test_lexer_colors_severity_by_shape_count(name):
    assert _first_token(name) is SEVERITY_TOKENS[WORDS['severities'][name]]


@pytest.mark.parametrize('name', WORDS['mixing'])
def test_lexer_colors_mixing_distribution_after_mixed(name):
    tokens = [(t, v) for t, v in AggLexer().get_tokens(f'mixed {name} 0.2')
              if v.strip()]
    assert tokens[1] == (Name.Function, name)


@pytest.mark.parametrize('name', WORDS['frequencies'])
def test_lexer_colors_frequency(name):
    assert _first_token(name) is Name.Function


@pytest.mark.parametrize('name', WORDS['declarations'])
def test_lexer_colors_declaration(name):
    assert _first_token(name) is Name.Label
