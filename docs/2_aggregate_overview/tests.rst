.. _testing:

#######################################
Testing ``aggregate``: a friendly guide
#######################################

This page explains how ``aggregate`` is tested: what the testing tool does, the
different *kinds* of test in the project, how to run them yourself, and how the
"golden" reference data is kept up to date. It assumes **no prior knowledge** of
Python testing.

.. contents:: On this page
   :local:
   :depth: 2


**************************
What is automated testing?
**************************

An *automated test* is a small piece of code that checks one fact about the
library and either passes (the fact holds) or fails (it doesn't). For example:

.. code-block:: python

   from aggregate import build

   def test_dice_mean_is_3point5():
       a = build('agg D dfreq [1] dsev [1:6]')   # one roll of a fair die
       assert abs(a.actual_m - 3.5) < 1e-12          # the mean should be 3.5

The magic word is ``assert``: "I assert that this is true." If it isn't, the
test fails and you find out *immediately* that a change broke something. A
project with hundreds of these tests is a safety net: you can refactor the
internals with confidence, because if you accidentally change an answer, a test
goes red.

Each test is just a Python function whose name starts with ``test_``, living in
a file whose name starts with ``test_`` in the ``tests/`` folder.


***************
What is pytest?
***************

`pytest <https://docs.pytest.org/>`_ is the program that **finds and runs** all
those ``test_*`` functions and reports which passed and which failed. You never
call the test functions yourself — pytest discovers them.

When you run pytest it:

1. scans the ``tests/`` folder for ``test_*.py`` files;
2. inside each, collects every ``test_*`` function;
3. runs them one by one;
4. prints a dot ``.`` for each pass and an ``F`` for each fail, then a summary
   like ``799 passed, 113 warnings in 58s``.

A failing test prints the line that failed and the actual vs expected values, so
you can see *what* went wrong without adding print statements.

The project's pytest settings live in ``pyproject.toml`` under
``[tool.pytest.ini_options]``:

.. code-block:: toml

   testpaths = ["tests"]            # where to look
   addopts   = "-ra --strict-markers"  # show a summary of non-passing tests

You don't need to understand those flags to use the tests.


*****************
Running the tests
*****************

Everything runs through ``uv`` (the environment manager). From the project root:

.. code-block:: bash

   # run the whole suite
   uv run pytest

   # run one file
   uv run pytest tests/test_negative_x.py

   # run one test by name (substring match with -k)
   uv run pytest -k dice

   # run one exact test
   uv run pytest tests/test_negative_x.py::test_fixed_n_closed_form

   # stop at the first failure (-x), and show local variables (-l)
   uv run pytest -x -l

   # quieter (-q) or more verbose (-v, lists every test name)
   uv run pytest -q
   uv run pytest -v

The full suite takes about a minute. During quick back-and-forth development it
is common to run just the relevant file (e.g. ``test_negative_x.py``) and only
run the whole suite before declaring a change finished.

.. note::

   On this Windows/PowerShell setup, set ``UV_LINK_MODE=copy`` if you invoke
   ``uv`` in a plain shell (``$env:UV_LINK_MODE = "copy"``). The Claude harness
   sets this automatically.


*************************
The kinds of test we have
*************************

The suite mixes several *styles* of test. Knowing which is which makes failures
much easier to interpret.

Unit tests — "does this one thing work?"
========================================

The bread and butter: build a small object, assert a known property. They are
fast and pinpoint exactly what broke.

* ``tests/test_moments.py`` — moment arithmetic (mean / CV / skew) for assorted
  builds.
* ``tests/test_style.py`` — the plotting style helpers return sensible
  dictionaries and set matplotlib options.
* ``tests/test_tweedie.py`` — the Tweedie distribution helpers.

Example (the spirit of many of these)::

   def test_value_type_member():
       a = build('agg VT 5 claims sev lognorm 100 cv 1 poisson')
       assert a.value_type == 'loss'      # default
       a.value_type = 'payoff'            # settable
       with pytest.raises(ValueError):
           a.value_type = 'nonsense'      # validated

``pytest.raises`` is how you assert that something *should* raise an error.

Parametrized tests — "do this for every case"
=============================================

Often you want the *same* check run over many inputs. ``@pytest.mark.parametrize``
turns one function into many test cases — pytest reports each separately.

.. code-block:: python

   @pytest.mark.parametrize('program', [
       'agg Id.A 10 claims sev lognorm 100 cv 2 poisson',
       'agg Id.B dfreq [3] dsev [1:6]',
   ])
   def test_default_path_identity(program):
       ...

That single function becomes two named cases:
``test_default_path_identity[agg Id.A ...]`` and ``[agg Id.B ...]``. The big
example is the **DecL grammar suite** (below).

The DecL grammar suite — every line of ``test_suite.agg``
=========================================================

``src/aggregate/agg/test_suite.agg`` is a long file of example DecL programs,
organised into categories A–O (frequencies, severities, reinsurance,
distortions, case studies, papers…). ``tests/test_decl_parser.py`` turns **each
line** into its own pair of parametrized tests:

* ``test_line_parses`` — the line parses to a valid ``(kind, name, spec)``
  triple (the program is grammatically legal).
* ``test_spec_matches_snapshot`` — the parsed ``spec`` matches a stored
  reference in ``tests/data/expected_specs.json``. This catches *semantic
  drift*: if a grammar change silently alters how a program is interpreted, the
  spec no longer matches and the test fails.

So adding a line to ``test_suite.agg`` automatically adds test coverage for it.

Snapshot / "golden master" tests — "does the answer still match the file?"
==========================================================================

A *snapshot* (or *golden master*) test compares today's computed output against
a previously-saved "known good" copy on disk. These guard the *numbers*, not
just "does it run".

* ``tests/test_baseline.py`` — the **characterization baseline**. It loads
  ``tests/baseline/data/manifest.json``, rebuilds a curated set of aggregates and
  portfolios at a **pinned grid** (fixed ``log2``/``bs`` so the answer is
  reproducible), recomputes their frames (``density_df``, ``describe``,
  ``stats_df`` …) and compares element-by-element against stored ``.parquet``
  files. It runs *every* case before reporting and collects all divergences into
  one summary, e.g.::

      Sym.Dice / density_df / p_total : max abs 3.4e-09 at row '7.0'

  This is the project's main "did I move a number I didn't mean to?" guard.
* ``tests/test_distortion_snapshot.py`` / ``test_severity_layer_golden.py`` —
  smaller golden-master checks for distortions and layered-severity moments.

The reference data is produced by *capture scripts* (``tests/baseline/capture.py``
and the ``tests/capture_*.py`` files). See
:ref:`Regenerating reference data <tests regenerating>`.

Feature suites — "this whole capability behaves correctly"
==========================================================

Larger files that exercise one feature end-to-end, usually mixing closed-form
checks, identities, and regressions:

* ``tests/test_negative_x.py`` — signed (profit/loss) severity and the output
  window. Includes exact closed-form checks (a fixed sum of ``{-2, 5}`` has known
  support ``{-6, 1, 8, 15}``), symmetry checks, the bucket/window estimator, and
  ``sev_density_df``.
* ``tests/test_reins_reporting.py`` / ``test_reins_buckets.py`` /
  ``test_reins_bivariate.py`` — reinsurance reporting frames, the rebucketing
  schemes, and the joint (ceded, net) bivariate law.
* ``tests/test_distortion_*`` — building, calibrating, and applying risk
  distortions.
* ``tests/test_bounds.py``, ``test_splice_suite.py``, ``test_underwriter.py`` —
  pricing bounds, spliced severities, and the ``build()`` entry point.

Error / robustness tests — "does it fail *gracefully*?"
=======================================================

* ``tests/test_parser_errors.py`` and ``test_parser_errors_integration.py`` —
  feed deliberately broken DecL and assert that the error message is clear and
  points at the right spot. Good error messages are a feature, so they are
  tested too.

Validation tests — "is the FFT answer trustworthy?"
===================================================

* ``tests/test_validation.py`` — the library's own self-check (``valid`` /
  ``explain_validation``) flags when the discretisation grid is too coarse
  (aliasing) or moments don't reconcile. These tests confirm the self-check
  fires when it should and stays quiet when the model is fine.


****************************
Fixtures and ``conftest.py``
****************************

A *fixture* is reusable setup shared by many tests, so each test doesn't repeat
it. Fixtures live in ``tests/conftest.py`` (a special file pytest loads
automatically) and are "requested" by naming them as a function argument:

.. code-block:: python

   def test_something(underwriter):     # 'underwriter' is a fixture
       ...                              # pytest builds it and passes it in

``aggregate``'s ``conftest.py`` provides two:

* ``test_suite_lines`` — all the preprocessed DecL lines from
  ``test_suite.agg``;
* ``underwriter`` — an ``Underwriter`` with those lines pre-loaded.

Both are ``scope="session"``: built once and reused across the whole run (fast).


************************************
Keeping ``decl-testers.agg`` in sync
************************************

When a hand-written test (e.g. ``test_negative_x.py``) uses DecL programs, the
*same* programs are mirrored into ``src/aggregate/agg/decl-testers.agg`` under a
matching section. That file is a human-readable catalogue of "programs the tests
rely on" — it is documentation, not itself auto-run, but it keeps the language
examples discoverable and in one place. Add to it whenever you add DecL-driven
tests.


.. _tests regenerating:

***************************
Regenerating reference data
***************************

Snapshot/golden tests compare against saved files. When you make a change that is
*supposed* to move the numbers (or move a column), you regenerate the reference —
**deliberately**, never automatically:

.. code-block:: bash

   uv run python tests/baseline/capture.py          # the characterization baseline
   uv run python tests/capture_severity_golden.py   # severity golden master
   uv run python tests/capture_distortion_snapshot.py
   # ... etc.

Then commit the regenerated ``.parquet`` / ``.json`` **in the same commit** as
the code change, with a message explaining *why* the numbers moved. The discipline
is: if a snapshot test fails, first decide whether the change was intended. If
yes, regenerate and explain. If no, you just caught a bug — fix the code, don't
regenerate.

.. warning::

   Regenerating bakes in *whatever* the code currently produces. Only do it once
   you have confirmed the *only* differences are the ones you intended (the
   baseline failure report lists each divergence, which makes this easy to
   check).


*****************
Reading a failure
*****************

A typical failure looks like::

   FAILED tests/test_negative_x.py::test_fixed_n_closed_form -
       assert {-6: 0.125, 1: 0.375, ...} == {-6: 0.125, 1: 0.374, ...}

pytest shows the assert that failed and both sides of the comparison. Workflow:

1. Read which test and which assertion.
2. Reproduce just that test: ``uv run pytest -k fixed_n_closed_form -l``.
3. Decide: bug in the code, or an intended change that needs the test / snapshot
   updated?

Tolerances matter: FFT results are exact only up to floating-point rounding, so
numeric tests compare with a small tolerance (``abs(x - y) < 1e-12``,
``np.allclose``, or the project's noise-aware helpers) rather than ``==``.


***************
Quick reference
***************

============================  ====================================================
Command                       Does
============================  ====================================================
``uv run pytest``             Run the whole suite
``uv run pytest -q``          …quietly (dots + summary)
``uv run pytest -k NAME``     Run tests whose name contains ``NAME``
``uv run pytest FILE``        Run one file
``uv run pytest FILE::TEST``  Run one exact test
``uv run pytest -x``          Stop at first failure
``uv run pytest -l``          Show local variables on failure
``uv run pytest -v``          List every test name
============================  ====================================================

Add a test whenever you add a feature or fix a bug (a test that would have caught
it). Run the relevant file as you work; run the whole suite before you finish.
