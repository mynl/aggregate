Parser
======

The DecL lexer and parser. DecL, the **Dec**\ laration **L**\ anguage, is the
domain-specific language users write to describe aggregates; see the
:doc:`DecL language reference <../4_dec_Language_Reference>` for the grammar and
worked examples. The grammar itself lives in ``aggregate/decl.lark`` (an Earley
grammar with a dynamic lexer, parsed with `Lark <https://lark-parser.readthedocs.io>`_);
the parser builds the ``(kind, name, spec)`` triple the
:class:`~aggregate.underwriter.Underwriter` turns into an object.

Most users never touch these classes directly, since :func:`~aggregate.build`
drives them, but they are public for tooling and introspection.

.. currentmodule:: aggregate.parser

.. autosummary::

   UnderwritingLexer
   UnderwritingParser
   grammar
   INHERIT_PREMIUM

.. currentmodule:: aggregate.decl_writer

.. autosummary::

   format_program
   spec_to_decl

.. currentmodule:: aggregate.parser_errors

.. autosummary::

   ErrorReport
   format_error

.. currentmodule:: aggregate.decl_pygments

.. autosummary::

   AggLexer

.. currentmodule:: aggregate.parser

Lexer
-----

.. autoclass:: aggregate.parser.UnderwritingLexer

Parser
------

.. autoclass:: aggregate.parser.UnderwritingParser

Grammar helper
--------------

.. autofunction:: aggregate.parser.grammar

Premium inheritance sentinel
----------------------------

.. autodata:: aggregate.parser.INHERIT_PREMIUM
   :no-value:

Unparser
--------

The inverse direction: a parsed ``spec`` rendered back to canonical DecL.
:func:`~aggregate.decl_writer.format_program` and
:func:`~aggregate.decl_writer.spec_to_decl` are re-exported at the top level,
and every DecL-created object exposes the round trip as ``.pprogram`` /
``.format_program()`` (see :class:`aggregate._program.ProgramMixin`).

.. automodule:: aggregate.decl_writer

Parse errors
------------

The structured report carried on a failed :func:`~aggregate.build`'s
``ValueError.report``; see :ref:`4_dec_Language_Reference:Reading Parse Errors` in the language
reference for the usage patterns.

.. automodule:: aggregate.parser_errors

Syntax highlighting
-------------------

:class:`~aggregate.decl_pygments.AggLexer` is a Pygments lexer for DecL, a
hand-written mirror of ``decl.lark``. Pygments resolves it globally through an
entry point, so ``:language: agg`` in these docs, ``pygmentize`` on a ``.agg``
file, and the colorized :func:`~aggregate.decl_writer.format_program` output all
reach the same class.
Because the mirror is hand-written it can drift from the grammar;
``tests/test_grammar_sync.py`` is the guard.

.. automodule:: aggregate.decl_pygments
