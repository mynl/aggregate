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
object. It also carries how the entry describes itself, taken from its DecL
trailer: a one-line ``note{...}``, ``tags{...}`` for grouping, and
``hints{...}`` for build settings.
:meth:`~aggregate.underwriter.Underwriter.recipe` returns one by name;
:attr:`~aggregate.underwriter.Underwriter.recipes` is the frame over all of
them; :meth:`~aggregate.underwriter.Underwriter.discover` filters by name, kind
or tag.

One recipe base can serve many callers.
:meth:`~aggregate.underwriter.Underwriter.fork` returns an isolated copy
sharing the parsed entries, so a notebook gets a scratch base and a multi-user
host gets one namespace per user for the cost of a dict copy;
:meth:`~aggregate.underwriter.Underwriter.preview` reports what a program would
declare and what it leans on, with the provenance of each referent, without
building anything.

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
   RecipeNotFound
   ProgramPreview
   ResolvedReference

Recipe
------

.. autoclass:: aggregate.recipe.Recipe

.. automodule:: aggregate.recipe
   :exclude-members: Recipe

Underwriter class
-----------------

.. autoclass:: aggregate.underwriter.Underwriter

Module functions
----------------

.. automodule:: aggregate.underwriter
   :exclude-members: Underwriter
