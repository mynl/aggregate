Parser
======

The DecL lexer and parser. DecL — the **Dec**\ laration **L**\ anguage — is the
domain-specific language users write to describe aggregates; see the
:doc:`DecL language reference <../4_dec_Language_Reference>` for the grammar and
worked examples. The grammar itself lives in ``aggregate/decl.lark`` (an Earley
grammar with a dynamic lexer, parsed with `Lark <https://lark-parser.readthedocs.io>`_);
the parser builds the ``(kind, name, spec)`` triple the
:class:`~aggregate.underwriter.Underwriter` turns into an object.

Most users never touch these classes directly — :func:`~aggregate.build` drives
them — but they are public for tooling and introspection.

.. currentmodule:: aggregate.parser

.. autosummary::

   UnderwritingLexer
   UnderwritingParser
   grammar

Lexer
-----

.. autoclass:: aggregate.parser.UnderwritingLexer

Parser
------

.. autoclass:: aggregate.parser.UnderwritingParser

Grammar helper
--------------

.. autofunction:: aggregate.parser.grammar
