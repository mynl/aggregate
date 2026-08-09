"""Permanent regression net for the shipped DecL library.

``library.agg`` is the one shipped library and the default recipe base. It
replaced the overlapping ``examples`` / ``cookbook`` /
``actuarial-severity-curves`` trio at 1.0.0a159 ([Recipe-Library], phase 3 of
``dev/plan-meta-data.md``).

Every statement must parse to a valid ``(kind, name, spec)`` and any intra-file
builtin reference (``sev.UnitSeverity``, ``dist.PHDistortion``) must resolve in
load order -- ``Underwriter.load`` raises on either failure, so a successful
load is itself the assertion. The tests below add the invariants the merge
established: globally-unique names, and a tag on every entry.

This is the lasting net that lets the temporary ``_test_suite`` /
``_test_suite2`` migration scaffolding (and its SLY snapshot) retire before
beta: the user-facing library carries its own coverage here.

``decl-testers.agg`` is deliberately excluded -- it is the language-stress
corpus and includes intentional parser-error fixtures (section X), so it must
not load clean; it is exercised by ``test_decl_unparser`` and the mirrored
pytest cases.
"""
from collections import Counter

import pytest

from aggregate import Underwriter

# The user-facing shipped libraries (those meant to be wholly valid).
SHIPPED_LIBRARIES = ['library']


@pytest.fixture(scope='module')
def library():
    uw = Underwriter(databases='library')
    uw.load()
    return uw


@pytest.mark.parametrize('name', SHIPPED_LIBRARIES)
def test_shipped_library_loads(name):
    """Every statement in a shipped library parses and cross-resolves."""
    uw = Underwriter(databases=name)
    uw.load()
    assert len(uw._recipes) > 0, f'{name}.agg loaded no entries'


def test_library_names_are_unique_across_kinds(library):
    """``build('X')`` and ``build.recipe('X')`` must never be ambiguous.

    The recipe base is keyed ``(kind, name)``, so ``sev Pareto`` and
    ``agg Pareto`` could legally coexist -- and did, across the three files
    this library replaced. Giving that up is what lets a recipe be addressed
    by name alone. ``Underwriter._check_library_names_unique`` enforces it at
    load; this asserts the shipped file actually satisfies it.
    """
    counts = Counter(name for _kind, name in library._recipes)
    dupes = sorted(n for n, c in counts.items() if c > 1)
    assert not dupes, f'names used under more than one kind: {dupes}'


def test_library_retired_the_letter_prefixes(library):
    """Grouping is ``tags{}``; the old ``A.`` / ``K.`` filing codes are gone.

    Names leading with a citekey (``Mack2003.Lognorm``) are exempt: that is a
    citation, not a filing code, and it is the point of the entry.
    """
    import re
    citekey = re.compile(r'^[A-Z][a-zA-Z]+\d{4}[a-z]?\.')
    offenders = [n for _k, n in library._recipes
                 if re.match(r'^[A-Za-z]{1,3}\.', n) and not citekey.match(n)]
    assert not offenders, f'entries still carrying a filing prefix: {offenders}'


def test_every_library_entry_is_tagged(library):
    """Tags are the grouping mechanism, so an untagged entry is unreachable."""
    untagged = sorted(name for (_kind, name), pp in library._recipes.items()
                      if not pp.spec.get('tags'))
    assert not untagged, f'entries with no tags{{}}: {untagged}'


#: Tag namespaces. Everything is namespaced except one deliberate bare tag.
TAG_NAMESPACES = ('topic:', 'role:', 'check:')
#: ``slow`` names a pytest marker (``test_library_recipes`` maps it to
#: ``@pytest.mark.slow``), not a property of the subject, so namespacing it
#: would only add a translation step.
BARE_TAGS = {'slow'}


def test_tags_are_namespaced(library):
    """Every tag carries a namespace, or is the one allowed bare tag.

    This is the guard against the vocabulary regrowing type-restating tags.
    Before 1.0.0a161, 103 of 186 entries carried a tag that merely repeated
    their own kind (``aggregate`` on an ``agg``, ``portfolio`` on a ``port``),
    which said nothing ``discover(kind=...)`` does not already say and invited
    "why does type appear twice?". A namespaced tag cannot be confused for a
    kind, so ``sev UnitSeverity tags{topic:severity}`` is unambiguous.
    """
    offenders = {}
    for (kind, name), pp in library._recipes.items():
        bad = [t for t in pp.spec.get('tags', ())
               if not t.startswith(TAG_NAMESPACES) and t not in BARE_TAGS]
        if bad:
            offenders[f'{kind}:{name}'] = bad
    assert not offenders, (
        f'un-namespaced tags (use topic:/role:/check:, or add to BARE_TAGS '
        f'with a reason): {offenders}')


def test_no_tag_restates_its_own_kind(library):
    """Belt and braces: a tag must never just name the entry's own type.

    ``test_tags_are_namespaced`` already makes this impossible, but state the
    actual invariant separately -- the namespace rule is the *mechanism*, this
    is the *reason*, and a future bare tag added to ``BARE_TAGS`` must still
    satisfy it.
    """
    kind_words = {'agg': 'aggregate', 'sev': 'severity', 'port': 'portfolio',
                  'pnl': 'pnl', 'xpnl': 'pnl', 'bvagg': 'bivariate',
                  'distortion': 'distortion'}
    offenders = {f'{k}:{n}': kind_words[k]
                 for (k, n), pp in library._recipes.items()
                 if k in kind_words and kind_words[k] in pp.spec.get('tags', ())}
    assert not offenders, f'tags restating their own kind: {offenders}'


def test_library_is_the_default_recipe_base():
    """``build`` with no arguments reads library.agg."""
    from aggregate import build
    assert len(build.recipes) > 100
    a = build('LimitProfile')
    assert 'role:hero' in a.tags


# ----------------------------------------------------------------------
# Layout and trailer placement (1.0.0a178)
# ----------------------------------------------------------------------
#: Entries `decl_writer` cannot render back to what was written, so they are
#: hand-written and exempt from the canonical-layout check. Three causes, all
#: recorded as `[Unparser-Reference-Gaps]` in `dev/TODO.md`: a named object
#: reference (`sev.UnitSeverity`) is resolved and inlined at parse time with
#: nothing on the spec recording that a reference was written; the `minimum`
#: combinator drops its child distortion names; and the compact `ssev <c> -
#: <dist>` spelling renders in the general affine form `-1 * <dist> + <c>`,
#: whose leading `-1 *` parses two ways (see `tests/test_grammar_ambiguity.py`).
#: Shrinking this set is progress; growing it needs a reason.
#:
#: The `tweedie` clause was a fourth cause until 1.0.0a231, when it gained the
#: `_tweedie` provenance key: three entries left this set, which is what fixing
#: one of these gaps looks like.
UNPARSER_EXEMPT = {
    # named object reference
    'BernoulliFrequency', 'BinomialSimple', 'FixedFrequency',
    'GeometricFrequency', 'NegativeBinomialFrequency', 'NegativeBinomialMixed',
    'PoissonSimple', 'BasicMixedSev', 'InverseGaussianMixed',
    # named ENGINE reference: `xpnl USHurr ... less agg.USXOLTower`. Since
    # a216 [Inline-Port-Engine] an engine writes its body out, so the
    # canonical form inlines the whole referenced aggregate and the source's
    # one-line reference cannot be recovered from the spec. Same cause as the
    # group above; kept separate because the fix is different (the spec would
    # have to record that a reference was written).
    'USHurr',
    # distortion combinator
    'MinimumDistortion',
    # canonical form would be ambiguous
    'PremiumMinusLoss', 'SignedPremiumMinusLoss', 'PnLSignedSsev',
}


def test_library_is_written_in_the_canonical_layout():
    """Every non-exempt entry is byte-identical to what ``format_program`` renders.

    The shipped library is formatted by ``dev/done/reflow_library.py``, which is just
    ``format_program(..., layout='spread')`` over the file. Pinning that here
    makes the layout mechanical rather than a matter of taste, and makes a
    regeneration a no-op instead of a diff.
    """
    from aggregate.decl_writer import format_program
    from aggregate.parser import UnderwritingLexer

    uw = Underwriter(databases='library')
    uw.load()
    path = next(p for p in uw.databases if p.name == 'library.agg')
    offenders = []
    for statement in UnderwritingLexer.preprocess(path.read_text(encoding='utf-8')):
        kind, name, spec = uw.parser.parse(uw.lexer.tokenize(statement))
        if name in UNPARSER_EXEMPT:
            continue
        canonical = format_program((kind, name, spec), fmt='text',
                                   layout='spread', trailer=True)
        # Compare flattened: the file carries the layout, the check carries the
        # content, and preprocess is what turns one into the other.
        if UnderwritingLexer.preprocess(canonical)[0] != statement:
            offenders.append(name)
    assert not offenders, (
        f'{offenders} are not in canonical form -- run '
        f'`python dev/done/reflow_library.py`, or add a name to UNPARSER_EXEMPT '
        f'with a reason')


def test_no_library_port_binds_its_trailer_to_a_unit(library):
    """A ``port``'s ``note{}`` must sit on the port, not on its last unit.

    ``port_out`` places the trailer BEFORE ``agg_list``, so the portfolio's slot
    has closed by the time a unit is read: a note written after the last unit
    silently annotates that unit instead. The rule is deliberate and pinned by
    ``tests/test_trailer_attachment.py``; this asserts the shipped library is on
    the right side of it, since the mistake is invisible in the source.
    """
    offenders = {}
    for (kind, name), recipe in library._recipes.items():
        if kind != 'port':
            continue
        for unit in recipe.spec.get('spec', ()):
            # units are (kind, name, spec) triples
            unit_spec = unit[2] if isinstance(unit, tuple) and len(unit) == 3 else {}
            stray = [k for k in ('note', 'tags', 'hints') if unit_spec.get(k)]
            if stray:
                offenders[f'{name}.{unit_spec.get("name", "?")}'] = stray
    assert not offenders, (
        f'trailer bound to a portfolio unit instead of the portfolio: '
        f'{offenders}; move it to the port header line, after the name')


def test_duplicate_names_in_a_library_are_rejected(tmp_path):
    """The uniqueness rule is enforced at load, not just asserted here.

    Written against a synthetic ``library.agg`` in a temp dir so the guard is
    actually exercised -- a check that only ever sees valid input is not a
    check.
    """
    lib = tmp_path / 'library.agg'
    lib.write_text(
        'agg Duplicated 5 claims sev lognorm 10 cv 1 poisson tags{aggregate};\n'
        'sev Duplicated lognorm 10 cv 1 tags{severity};\n',
        encoding='utf-8')
    uw = Underwriter()
    with pytest.raises(ValueError, match='unique across kinds'):
        uw._read_file(lib)
