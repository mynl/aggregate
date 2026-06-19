"""Permanent smoke test for the shipped DecL libraries.

The bundled ``*.agg`` libraries that ship at v1.0 must load cleanly: every
statement parses to a valid ``(kind, name, spec)`` and any intra-file builtin
reference (``sev.One`` etc.) resolves in load order. ``Underwriter.load`` raises
on a parse error or an unresolved reference, so a successful load is the
assertion.

This is the lasting regression net that lets the temporary ``_test_suite`` /
``_test_suite2`` migration scaffolding (and its SLY snapshot) retire before beta:
the user-facing libraries carry their own coverage here.

``decl-testers.agg`` is deliberately excluded -- it is the language-stress corpus
and includes intentional parser-error fixtures (section X), so it must not load
clean; it is exercised by ``test_decl_unparser`` and the mirrored pytest cases.
"""
import pytest

from aggregate import Underwriter

# The user-facing shipped libraries (those meant to be wholly valid).
SHIPPED_LIBRARIES = ['examples', 'actuarial-severity-curves', 'cookbook']


@pytest.mark.parametrize('name', SHIPPED_LIBRARIES)
def test_shipped_library_loads(name):
    """Every statement in a shipped library parses and cross-resolves."""
    uw = Underwriter(databases=name)
    uw.load()
    assert len(uw._knowledge) > 0, f'{name}.agg loaded no entries'
