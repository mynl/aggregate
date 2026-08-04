Underwriter
===========

The :class:`~aggregate.underwriter.Underwriter` is the top-level entry point.
It owns the **recipe base** (the named ``agg`` / ``port`` / ``sev`` / ``distortion``
declarations loaded from the built-in and user databases), parses DecL
programs through the :doc:`parser <3_x_Parser>`, and constructs the
corresponding :class:`~aggregate.Aggregate` or :class:`~aggregate.Portfolio`
objects.

A **recipe** is one entry, whole: its kind and name, its parsed spec, the DecL
source it came from, where it was read from, and, once it has been built, the
object. It also carries the entry's own documentation, taken from its DecL
trailer: a one-line ``note{...}``, ``tags{...}`` for grouping, and, for the
cookbook-worthy few, a ``doc{{{...}}}`` holding a Problem / Solution /
Discussion / Check write-up that :meth:`~aggregate.recipe.Recipe.run` can
execute. :meth:`~aggregate.underwriter.Underwriter.recipe` returns one by name;
:attr:`~aggregate.underwriter.Underwriter.recipes` is the frame over all of
them; :meth:`~aggregate.underwriter.Underwriter.discover` filters by name, kind
or tag.

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

Recipe
------

.. autoclass:: aggregate.recipe.Recipe

.. automodule:: aggregate.recipe
   :exclude-members: Recipe

Cookbook
--------

The renderer half of the recipe surface: it turns a library's ``doc{{{...}}}``
entries into Quarto pages, so a library that documents and tests itself also
publishes itself. Point it at any :class:`Underwriter`, not just the shipped
library.

.. automodule:: aggregate.cookbook

Underwriter class
-----------------

.. autoclass:: aggregate.underwriter.Underwriter

Module functions
----------------

.. automodule:: aggregate.underwriter
   :exclude-members: Underwriter
