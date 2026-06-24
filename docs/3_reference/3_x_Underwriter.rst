Underwriter
===========

The :class:`~aggregate.underwriter.Underwriter` is the top-level entry point.
It owns the **knowledge base** (the named ``agg`` / ``port`` / ``sev`` / ``distortion``
specifications loaded from the built-in and user databases), parses DecL
programs through the :doc:`parser <3_x_Parser>`, and constructs the
corresponding :class:`~aggregate.Aggregate` or :class:`~aggregate.Portfolio`
objects.

The module-level :func:`~aggregate.underwriter.build` is the primary public
API: it wraps a default :class:`Underwriter` instance, so ``build('agg ...')``
parses a one-line program and returns a single updated object.
:func:`~aggregate.underwriter.build_many` is the multi-output form.

.. currentmodule:: aggregate.underwriter

.. autosummary::

   Underwriter
   build
   build_many
   CannotBuild

Underwriter class
-----------------

.. autoclass:: aggregate.underwriter.Underwriter

Module functions
----------------

.. automodule:: aggregate.underwriter
   :exclude-members: Underwriter
