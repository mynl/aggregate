.. _underwriter guide:

The Underwriter and Recipes
===========================

This section covers the :class:`Underwriter`, the object that creates everything else and keeps a library of named declarations. It shows how to write DecL statements and hand them to ``build``, the difference between building an object and looking one up, how the recipe base is filtered and audited, how a declaration carries a write-up that the docs render and the test suite executes, and how libraries are loaded and saved.

.. _uw what it is:

The Underwriter
---------------

:class:`Underwriter` is the interface to everything ``aggregate`` computes. It does three things:

#. Creates objects from DecL programs (:doc:`../4_dec_Language_Reference`).
#. Keeps a library of named DecL declarations, the **recipe base**. Anything you build is added to it automatically.
#. Describes, tests and audits that library, because a declaration carries its own documentation in its DecL trailer.

The third is new at 1.0.0 and is the subject of :ref:`uw recipes`. A library entry can hold a worked write-up that the docs render and the test suite executes, so an entry documents and tests itself.

To get started, import ``build``, a pre-configured :class:`Underwriter`, and :func:`qd`, the quick-display function.

.. ipython:: python
    :okwarning:

    from aggregate import build, qd
    import pandas as pd, numpy as np

Printing ``build`` reports its name, how many recipes it holds, the default hyper-parameters, and where it reads from. ``user_dir`` (``~/.aggregate``) is where your own databases live; ``default_dir`` holds the package's own.

.. ipython:: python
    :okwarning:

    build

.. _uw create:

Creating objects with DecL
--------------------------

To build an :class:`Aggregate` and report key statistics for frequency, severity and aggregate takes two commands.

.. ipython:: python
    :okwarning:

    a01 = build('agg Guide:01 100 claims 100 xs 0 sev lognorm 10 cv 1.25 poisson')
    qd(a01)

DecL is meant to be readable, so you can probably guess what that says: 100 expected claims, each a lognormal with mean 10 and CV 1.25 limited to the layer 100 xs 0, with Poisson claim count. The units are thousands.

.. important::

    A DecL statement is one logical line.

Statements are separated by a blank line or by a semicolon at end of line. Every other newline is just whitespace, so one statement may span as many physical lines, at whatever indentation, as you like. That is the layout to prefer for anything longer than the example above:

.. ipython:: python
    :okwarning:

    a02 = build('''
        agg Guide:02
            100 claims
            100 xs 0
            sev lognorm 10 cv 1.25
            poisson
    ''')
    qd(a02)

Comments (``#`` or ``//``) are transparent: a full-line comment between the clauses of one statement simply vanishes and the lines around it stay in the same statement. The corollary is that a comment cannot separate two statements. Use a blank line or a ``;``.

If you prefer Python string concatenation, mind the trailing spaces::

    a01 = build('agg Guide:01 '        # note the space before each closing quote
                '100 claims '
                '100 xs 0 '
                'sev lognorm 10 cv 1.25 '
                'poisson')

Without them the parts run together (``...Guide:01100 claims...``) and you get a parse error.

.. _uw three patterns:

Comparing ``build(x)`` and ``build.recipe(x)``
----------------------------------------------

There is an important distinction between ``build(x)`` and ``build.recipe(x)``. Both take the same string argument but they return different things.

===================  =====================  =======================================================
Call                 Returns                What it does
===================  =====================  =======================================================
``build(x)``         the live object        Parses ``x`` as DecL, or looks it up by name, then
                                            constructs and updates it. The one you want most of
                                            the time.
``build.recipe(x)``  a :class:`Recipe`      Looks up only. Never constructs, never parses.
                                            ``.object`` is always ``None``. Also spelled
                                            ``build[x]``, the subscript form.
===================  =====================  =======================================================

.. ipython:: python
    :okwarning:

    obj = build('ThreeDice')            # constructs
    rec = build.recipe('ThreeDice')     # looks up
    type(obj).__name__, type(rec).__name__, rec.object is None

The difference is construct against look up, and it is deliberate. The recipe base stores DecL specs, which are small and picklable, not live objects. Objects are made on demand for two reasons: a :class:`Portfolio` needs an :class:`Underwriter` reference that may differ between sessions, and each ``build('X')`` hands back a fresh instance so one caller's ``update()`` cannot bleed into another's.

.. ipython:: python
    :okwarning:

    build('ThreeDice') is build('ThreeDice')     # a fresh object every time

``build.recipe(x)`` takes a name and an optional ``kind=``, needed only when a name is not unique across kinds. The subscript takes the same two forms, ``build[name]`` and ``build[kind, name]``, and delegates to ``recipe``, so it raises the identical :class:`KeyError` on a name that matches zero or more than one entry.

.. ipython:: python
    :okwarning:

    build.recipe('ThreeDice', kind='agg').name
    build['agg', 'ThreeDice'].kind

For a program with more than one top-level output, ``build`` raises and points you at :meth:`build_many`, which always returns the full ``list[Recipe]`` with ``.object`` populated on each.

.. ipython:: python
    :okwarning:

    rv = build.build_many('agg Guide:A dfreq [1] dsev [1:6]\n\n'
                          'agg Guide:B dfreq [2] dsev [1:6]')
    [(r.kind, r.name, type(r.object).__name__) for r in rv]

A spec that parses but cannot stand alone raises :class:`CannotBuild`. The usual case is a named mixture severity, which only exists inside an :class:`Aggregate`; use :meth:`build_many` and read ``.spec`` directly.

.. _uw recipe base:

The recipe base
---------------

Everything ``build`` knows lives in one flat store keyed ``(kind, name)``, fed by the ``.agg`` files it loads and by anything you build in the session. :attr:`build.recipes` is the DataFrame view.

.. ipython:: python
    :okwarning:

    qd(build.recipes.iloc[:5, :3], justify='left', max_colwidth=60)

The columns divide into identity, description, and the payload:

``note``, ``tags``
    How the entry describes itself. ``note`` is a boolean; ``tags`` is the tuple of slugs.
``source``
    Which ``.agg`` file it came from, or ``'session'``.
``program``, ``spec``
    The DecL statement and the parsed constructor kwargs. Wide, so they sit last; ``.iloc[:, :3]`` is the readable slice.

The question it exists to answer:

.. ipython:: python
    :okwarning:

    len(build.recipes.query('not note'))   # entries with nothing said about them

.. _uw discover:

Finding things with :meth:`discover`
------------------------------------

:meth:`discover` filters the recipe base on three independent axes and, if you ask, builds each match.

.. ipython:: python
    :okwarning:

    build.discover('Dice')                      # regex on the NAME
    build.discover(kind='sev').head()           # by TYPE
    build.discover(tags='role:hero')            # by SUBJECT

``kind`` is the type filter and ``tags`` the subject filter, and they are independent on purpose. Library tags never restate an entry's own kind, because a ``severity`` tag on a ``sev`` entry would say nothing ``kind='sev'`` does not. An ``agg`` that demonstrates a severity form is therefore ``kind='agg'`` and ``tags='topic:severity'``:

.. ipython:: python
    :okwarning:

    build.discover(kind='agg', tags='topic:severity').head()

Tags narrow rather than widen: an entry matches when it carries every tag given. The shipped vocabulary is namespaced.

``topic:``
    What the entry is about: ``severity`` ``frequency`` ``aggregate`` ``reinsurance`` ``pnl`` ``portfolio`` ``distortion`` ``bivariate`` ``numerics``.
``role:``
    Where it stands: ``hero`` (landing page), ``intro`` (teaching order), ``reference``, ``paper`` (reproduces a published result).
``check:``
    Which invariant its Check asserts: ``reconciliation`` ``scaling-sweep`` ``independent-oracle`` ``limiting-case`` ``round-trip`` ``cross-object``.

Pass ``describe=True`` or ``plot=True`` to build every match and show it, and ``return_objects=True`` to get the objects back alongside the frame. Be careful: that builds everything the filter matched.

.. _uw recipes:

Recipes: describe, test, audit
------------------------------

A DecL statement can carry three **trailer** clauses. They are order-free and at most one of each.

.. code-block:: text

    note{...}       one line. What this is.
    tags{...}       comma or space separated slugs. Grouping and selection.
    hints{...}      key=value; build settings, e.g. hints{log2=18; bs=1/64}

Each states a fact about the entry, and a ``note`` is the norm: it is all :meth:`discover` and an object picker need. There is no second tier to fill in, so ``build.recipes`` is a directory of the library rather than a backlog.

:meth:`build.recipe` returns the entry as a :class:`~aggregate.recipe.Recipe`, the record the recipe base stores:

.. ipython:: python
    :okwarning:

    r = build.recipe('LimitProfile')
    r
    r.note, r.tags, r.hints
    print(r.decl)

:attr:`~aggregate.recipe.Recipe.decl` is the useful one. It is the declaration re-rendered canonically from the spec, carrying ``hints`` and nothing else, since hints change how the object builds and the rest of the trailer is about the entry rather than part of it. Printing it is how a document elsewhere quotes a library program without retyping it, and because it reads the live entry it cannot go stale.

.. note::

   A fourth clause, ``doc{{{ ... }}}``, carried a long-form Problem / Solution /
   Discussion / Check write-up here until 1.0.0a301, executed by a pytest
   harness and rendered into a Quarto cookbook. It was retired before 1.0: a
   trailer states facts about an entry and a book is not one of them, and the
   DecL stability promise would have frozen the arrangement for the life of the
   major version. The write-ups became standalone notes, and the invariants
   they asserted became ordinary asserts in
   ``tests/test_library_entries.py``, which also checks that every shipped
   entry builds.

.. _uw program:

``program`` versus ``pprogram``
-------------------------------

Every DecL-created object carries both a ``program`` and a ``pprogram`` attribute.

.. ipython:: python
    :okwarning:

    d = build('''
        agg Guide:03
            dfreq [1:3]
            dsev [1:6]
            note{a stored note}
            hints{log2=12}
    ''', update=False)
    d.program                  # what the parser was handed
    print(d.pprogram)          # what it understood

:attr:`program` is the statement as the parser received it, not what you
typed. It has been folded onto one line whatever its source layout, had its comments stripped, kept a few double spaces from the bracket step, and had any ``doc`` body replaced by URL-safe base64. That last one surprises people:

.. ipython:: python
    :okwarning:

    build.recipe('ThreeDice').program[-34:]      # the tail of a documented entry

The encoding is deliberate rather than corruption. It happens first, which is what lets a doc body carry ``#`` headings, blank lines and fenced code through the later preprocessing steps. Read it back decoded with ``.doc``.

:attr:`pprogram` is that statement re-parsed and rendered back from the spec, so it is canonical. Equivalent declarations collapse to one form, and what you see is the parse. Four things you will notice:

.. ipython:: python
    :okwarning:

    # range sugar expands
    print(build('''
        agg G:a
            dfreq [1:3]
            dsev [1:6]
    ''', update=False).pprogram)

.. ipython:: python
    :okwarning:

    # a builtin reference is resolved inline
    print(build('''
        agg G:b
            5 claims
            sev sev.UnitSeverity
            fixed
    ''', update=False).pprogram)

.. ipython:: python
    :okwarning:

    # an expression is evaluated
    print(build('''
        agg G:c
            10 claims
            sev lognorm 10 cv exp(.5)
            poisson
    ''', update=False).pprogram)

.. ipython:: python
    :okwarning:

    # `po` (part of) normalizes to the equivalent `so` (share of)
    print(build('''
        agg G:d
            10 claims
            100 xs 0
            sev lognorm 10 cv 1
            occurrence net of 50 po 100 xs 0
            poisson
    ''', update=False).pprogram)

Comparing the two is a quick way to see what the parser actually understood, which is not always what you thought you wrote. Neither is the source file; for that, read the ``.agg``.

:meth:`format_program` exposes the axes. It omits the trailer by default, because formatting a program is nearly always about the math and the insurance, not the metadata around it.

.. ipython:: python
    :okwarning:

    print(d.format_program(trailer=True))          # all four clauses
    print(d.format_program(layout='terse'))        # one line
    print(d.format_program(trailer=('hints',)))    # only what changes the build

.. _uw databases:

Loading and saving
------------------

``build`` reads the databases named in your config; a bare :class:`Underwriter` loads nothing, which is what you want for a test fixture or an isolated library.

.. ipython:: python
    :okwarning:

    from aggregate import Underwriter
    uw = Underwriter(databases='library')
    uw.load()
    len(uw.recipes)

:meth:`load` is the one load verb. It takes a filename, a glob, one of the
reserved names ``'default'``, ``'user'`` or ``'all'``, or a list of those, and reads additively. Files are resolved against ``cwd``, then ``user_dir``, then ``default_dir``.

.. ipython:: python
    :okwarning:

    build.available_databases()          # what could I load?
    build.resolve_databases('library')   # what would this request load?
    build.databases                      # what did I actually load?

:meth:`reload` resets to the as-created state and re-reads from disk, dropping
session entries and picking up on-disk edits.

Saving is an explicit export, because the recipe base has no "active" database. :meth:`to_agg` writes a selection to a ``.agg`` file that re-loads cleanly; entries are written in dependency order so named references resolve. The default writes everything built this session to ``~/.aggregate/<name>.agg``::

    build.to_agg('mybook')                        # source='session' by default
    build.to_agg('sevs', kind='sev', source='all', mode='w')

.. _uw behind:

Behind the scenes
-----------------

Skip this the first time through.

Every declaration has a **kind** and a **name**, and exists in up to four forms. A :class:`Recipe` carries all of them:

``kind``
    ``sev`` :class:`Severity`, ``agg`` :class:`Aggregate`, ``port`` :class:`Portfolio`, ``distortion`` :class:`Distortion`, plus ``pnl``, ``xpnl`` and ``bvagg``.
``name``
    Yours, given in the program. Different from the Python variable holding the object.
``spec``
    The parsed dictionary of constructor kwargs.
``program``
    The DecL statement as parsed.
``object``
    The live instance, once the factory has run. ``None`` from a lookup.

.. ipython:: python
    :okwarning:

    r = build.recipe('LimitProfile')
    r.kind, r.name, sorted(r.spec)[:6], r.object is None

Names in the shipped ``library.agg`` are globally unique across kinds, enforced at load, so ``build('X')`` and ``build.recipe('X')`` always mean the same entry and no ``kind=`` is ever needed. The store itself allows ``sev Pareto`` and ``agg Pareto`` to coexist; the shipped library gives that up on purpose.

For parser debugging, :meth:`interpret_file` runs every statement in a ``.agg`` file through the parser without constructing anything, returning per-statement error information. It splits the file the way :meth:`load` does, so a statement laid out over several lines is one row.

Configuration
~~~~~~~~~~~~~

Defaults come from ``~/.aggregate/config.toml``. :meth:`show_settings` prints every resolved setting and where its value came from.

.. ipython:: python
    :okwarning:

    build.show_settings()

.. _uw cheat:

One-page summary
----------------

=====================================  =========================================================
Call                                   Gives you
=====================================  =========================================================
``build('agg X ...')``                 Parse, register, construct, update. The live object.
``build('X')``                         Look up by name, construct. A fresh object each call.
``build.recipe('X')``                  The :class:`Recipe`. No construction.
``build.build_many(prog)``             ``list[Recipe]``, ``.object`` populated on each.
``build.recipes``                      The whole library as a DataFrame, plus its doc audit.
``build.discover(rx, kind=, tags=)``   Filter by name, type, subject. Optionally build and plot.
``build.recipe('X').run()``            Execute the entry's Solution and Check.
``a.program`` / ``a.pprogram``         As parsed / canonically re-rendered.
``a.format_program(trailer=True)``     Full control of markup, layout and trailer.
``build.load(...)``                    Read more.
``build.to_agg(...)``                  Export a selection.
=====================================  =========================================================
