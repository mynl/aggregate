"""A builtin reference keeps the referenced entry's stored trailer [Reference-Trailer-Preservation].

``agg.NAME`` stands for the stored entry, so the note, tags and hints that
entry declares are part of what the reference names. Before 1.0.0a318 the
parser splatted the whole trailer dict over the stored spec, and ``trailer``
seeds ``note`` and ``hints`` with empty strings, so an absent outer clause
blanked the stored values. The visible cost was the grid: every hinted library
entry silently rebuilt on the auto sized grid instead of the one its author
pinned, and ``agg.MED.WithPicks`` failed outright, since its picks attachments
lie on the grid only at the pinned ``bs=125``.

The rule these tests pin: **outer wins where written, stored survives where
not.** Tags already behaved that way, having no seeded default, which is what
marked the note and hints clobber as accidental.

Each test takes its own ``build.fork()`` (a private recipe base over the shared
parsed entries, 1.0.0a302). That is not tidiness. A build re-registers its
entry, so a reference carrying an explicit outer ``hints{}`` legitimately
stores those hints back, and a second case reading the same name in the same
session would see the first case's grid rather than the entry's own. That is
``[Session-Build-Clobbers-The-Trailer]`` in ``dev/TODO.md``, which is a
separate open item; forking keeps it out of these assertions.

The source program is mirrored in ``src/aggregate/agg/decl-testers.agg`` under
the AE section. The *reference* statements are deliberately not mirrored: the
unparser resolves a reference back to the full spec rather than to
``agg.NAME``, so they are not round-trip fixed points
(``[Unparser-Reference-Gaps]``).
"""

import pytest

from aggregate import build

#: The stored entry every reference below points at. ``bs=125`` is chosen to be
#: non dyadic, so a reference that loses the hints lands on a visibly different
#: grid rather than accidentally agreeing with the auto sized one.
SOURCE = ('agg AE.Trailer.RefSource 1 claim sev lognorm 100 cv 2 fixed '
          'note{the stored trailer a reference must preserve} '
          'tags{topic:reference} '
          'hints{bs=125; log2=18}')


@pytest.fixture
def uw():
    """A private recipe base carrying the source entry and nothing else."""
    u = build.fork()
    u(SOURCE)
    return u


def test_reference_preserves_the_stored_trailer(uw):
    """A bare reference builds on the entry's pinned grid, with its note intact."""
    a = uw('agg.AE.Trailer.RefSource')
    assert a.bs == 125.0
    assert a.log2 == 18
    assert a.spec['note'] == 'the stored trailer a reference must preserve'


def test_reference_preserves_the_stored_tags(uw):
    """Tags were never clobbered; assert it so the fix cannot regress them."""
    a = uw('agg.AE.Trailer.RefSource')
    assert 'topic:reference' in a.spec['tags']


def test_an_outer_hints_clause_still_overrides(uw):
    """Written outer hints win, and the unstated note still comes from the entry."""
    a = uw('agg.AE.Trailer.RefSource hints{bs=250; log2=17}')
    assert a.bs == 250.0
    assert a.log2 == 17
    assert a.spec['note'] == 'the stored trailer a reference must preserve'


def test_an_outer_note_overrides_while_the_stored_hints_survive(uw):
    """The two keys are independent: overriding one does not drop the other."""
    a = uw('agg.AE.Trailer.RefSource note{outer note}')
    assert a.spec['note'] == 'outer note'
    assert a.bs == 125.0
    assert a.log2 == 18


def test_building_a_reference_leaves_the_stored_entry_alone(uw):
    """The reference build must not write a blanked trailer back to the base.

    The clobber had a second life: the reference's spec was re-registered under
    the same name, so the entry stayed broken for the rest of the session and a
    later build by bare name also came out on the wrong grid.
    """
    uw('agg.AE.Trailer.RefSource')
    stored = uw._recipes[('agg', 'AE.Trailer.RefSource')].spec
    assert stored['hints'] == 'bs=125; log2=18'
    assert stored['note'] == 'the stored trailer a reference must preserve'
    assert uw('AE.Trailer.RefSource').bs == 125.0


def test_hinted_library_entry_with_picks_builds_through_a_reference():
    """The reported crash: ``agg.MED.WithPicks`` off its pinned grid.

    Pinned on the shipped entry deliberately. The picks attachments are exact
    multiples of the entry's ``bs=125`` and of nothing the auto sizer picks, so
    this is the end to end case that the trailer really did reach the update.
    """
    a = build.fork()('agg.MED.WithPicks')
    assert a.bs == 125.0
    assert a.log2 == 18
