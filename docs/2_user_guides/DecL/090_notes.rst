.. _2_agg_class_note_clause:

.. reviewed 2022-12-24
.. updated 2026-06-03 (note/hints split, 1.0.0a25)

The Note and Hints Clauses
-----------------------------

A program can carry two optional trailing clauses, in either order (at most one
of each): a ``note{...}`` free-text annotation and a ``hints{...}`` clause of
build settings. Both are allowed wherever a note was previously permitted
(``agg``, ``sev``, ``port``). The text is enclosed in braces and cannot include
a line break.

The note clause
~~~~~~~~~~~~~~~~~

``note{...}`` is **pure free-text** — a human-readable comment carried on the
object. It has no effect on the computation.

::

    note{US Prems Ops, light hazard severity; for ABC account}

The hints clause
~~~~~~~~~~~~~~~~~~

``hints{...}`` carries build/update settings as ``key=value;`` pairs. The
recognised keys are the build knobs ``log2``, ``bs``, ``padding``,
``normalize``, ``recommend_p``, ``x_min``, ``x_max``, ``sev_calc``,
``discretization_calc``, ``force_severity``.

::

    agg MyBook 100 claims sev lognorm 100 cv 2 poisson hints{log2=16; bs=1/32}

Values are inferred generically: integer, float, an ``a/b`` fraction (so
``bs=1/64`` works), ``True``/``False``, or otherwise a string.

- **Caller always wins.** Explicit keyword arguments passed to ``build(...)``
  override the in-program ``hints`` — including ``recommend_p``.
- **Forgiving.** An unknown key is warned about and dropped; a duplicate key
  warns and the last value wins; a malformed clause warns and is skipped. A bad
  hint never crashes the build.

.. note::

    Before 1.0.0a25, ``note{...}`` doubled as the settings side-channel (e.g.
    ``note{...; log2=16}``). That is deprecated: settings must now go in
    ``hints{}``. A note that still looks like it carries ``key=value`` settings
    emits a warning and is otherwise treated as pure text.
