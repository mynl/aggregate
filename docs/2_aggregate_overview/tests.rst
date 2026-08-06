.. _testing:

Testing ``aggregate``
=====================

This section explains how ``aggregate`` is tested: what the testing tool does, the different kinds of test in the project, how to run them yourself, and how the golden reference data is kept up to date. It assumes no prior knowledge of Python testing.

What is automated testing?
--------------------------

An automated test is a small piece of code that checks one fact about the library and either passes, when the fact holds, or fails, when it does not. For example:

.. code-block:: python

   from aggregate import build

   def test_dice_mean_is_3point5():
       a = build('agg D dfreq [1] dsev [1:6]')   # one roll of a fair die
       assert abs(a.actual_m - 3.5) < 1e-12      # the mean should be 3.5

The magic word is ``assert``: "I assert that this is true." If it is not, the test fails and you find out immediately that a change broke something. A project with hundreds of these tests is a safety net: you can refactor the internals with confidence, because if you accidentally change an answer, a test goes red.

Each test is a Python function whose name starts with ``test_``, living in a file whose name starts with ``test_`` in the ``tests/`` folder.

What is pytest?
---------------

`pytest <https://docs.pytest.org/>`_ is the program that finds and runs all those ``test_*`` functions and reports which passed and which failed. You never call the test functions yourself, because pytest discovers them.

When you run pytest it:

1. scans the ``tests/`` folder for ``test_*.py`` files;
2. inside each, collects every ``test_*`` function;
3. runs them, spread across your cores;
4. prints a dot for each pass and an ``F`` for each fail, then a summary line.

A failing test prints the line that failed and the actual against expected values, so you can see what went wrong without adding print statements.

The project's pytest settings live in ``pyproject.toml`` under ``[tool.pytest.ini_options]``:

.. code-block:: toml

   testpaths = ["tests"]
   addopts = "-ra --strict-markers -n auto --dist loadgroup -m 'not slow'"

Those four flags are worth knowing, because they shape every run:

``-ra``
    summarize the non-passing tests at the end.
``--strict-markers``
    an unregistered marker is an error, so a typo in a ``@pytest.mark.`` name
    cannot silently do nothing. Markers are registered in the same
    ``pyproject.toml`` section.
``-n auto``
    fan the suite across all cores, using ``pytest-xdist``. The workload is
    FFT and numpy bound and parallelizes cleanly. Disable it with ``-n0`` when
    you need a breakpoint or deterministic single-process ordering.
``--dist loadgroup``
    ungrouped tests are distributed as usual, but every test sharing an
    ``xdist_group`` name goes to one worker, so those tests run sequentially
    against each other. See :ref:`tests markers`.
``-m 'not slow'``
    the fast-by-default local loop. See :ref:`tests markers`.

Running the tests
-----------------

Everything runs through ``uv``, the environment manager. From the project root:

.. code-block:: bash

   # the fast suite: everything except the `slow` cases
   uv run pytest

   # absolutely everything, including the quarantined heavy cases
   uv run pytest -m 'slow or not slow'

   # just the heavy cases
   uv run pytest -m slow

   # run one file
   uv run pytest tests/test_negative_x.py

   # run one test by name (substring match with -k)
   uv run pytest -k dice

   # run one exact test
   uv run pytest tests/test_negative_x.py::test_fixed_n_closed_form

   # stop at the first failure (-x), and show local variables (-l)
   uv run pytest -x -l

   # quieter (-q) or more verbose (-v, which lists every test name)
   uv run pytest -q
   uv run pytest -v

The fast suite is a couple of thousand cases and takes a minute or two on a modern machine. The full suite, with the ``slow`` cases included, takes several minutes, so it belongs at a commit boundary rather than in the edit loop. During quick back-and-forth development, run the relevant file and leave the whole suite until you are declaring a change finished.

.. note::

   On a Windows or PowerShell setup, set ``UV_LINK_MODE=copy`` if you invoke ``uv`` in a plain shell (``$env:UV_LINK_MODE = "copy"``).

.. _tests markers:

Markers: ``slow`` and ``xdist_group``
-------------------------------------

Two markers keep the everyday run fast and the heavy run possible. Both are registered in ``pyproject.toml``, and ``--strict-markers`` means an unregistered one is an error rather than a no-op.

``slow``
    Bleeding-edge, multi-minute cases. The three bivariate suites
    (``test_bivariate.py``, ``test_massive_bivariate.py``,
    ``test_reins_bivariate.py``) are tagged at module level with
    ``pytestmark = pytest.mark.slow``; between them they hold the cases that
    set the whole suite's wall-clock floor. ``addopts`` carries
    ``-m 'not slow'``, so the everyday ``uv run pytest`` skips them. Quarantine
    a new expensive case with ``@pytest.mark.slow``, and find candidates with
    ``uv run pytest -m 'slow or not slow' --durations=20``.
``xdist_group``
    About memory, not time. Tests sharing a group name go to one worker and so
    run sequentially against each other. ``test_bivariate.py`` uses it: each
    case allocates a 2-D FFT grid, and several running concurrently once
    exhausted memory, surfacing as a numpy allocation failure in a test whose
    own grid was only 64 by 64. The pressure came from its neighbors. Reach for
    this marker when tests are individually fine but collectively too large,
    and for ``slow`` when a test is simply long.

The kinds of test we have
-------------------------

The suite mixes several styles of test. Knowing which is which makes failures much easier to interpret.

Unit tests, "does this one thing work?"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bread and butter: build a small object, assert a known property. They are fast and pinpoint exactly what broke.

* ``tests/test_moments.py``, moment arithmetic (mean, CV, skewness) for assorted builds.
* ``tests/test_style.py``, the plotting style helpers return sensible dictionaries and set matplotlib options.
* ``tests/test_tweedie.py``, the Tweedie distribution helpers.

Here is the spirit of many of them::

   def test_value_type_member():
       a = build('agg VT 5 claims sev lognorm 100 cv 1 poisson')
       assert a.value_type == 'loss'      # default
       a.value_type = 'payoff'            # settable
       with pytest.raises(ValueError):
           a.value_type = 'nonsense'      # validated

``pytest.raises`` is how you assert that something should raise an error.

Parametrized tests, "do this for every case"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often you want the same check run over many inputs. ``@pytest.mark.parametrize`` turns one function into many test cases, and pytest reports each separately.

.. code-block:: python

   @pytest.mark.parametrize('program', [
       'agg Id.A 10 claims sev lognorm 100 cv 2 poisson',
       'agg Id.B dfreq [3] dsev [1:6]',
   ])
   def test_default_path_identity(program):
       ...

That single function becomes two named cases, ``test_default_path_identity[agg Id.A ...]`` and ``[agg Id.B ...]``. The big example is :ref:`tests grammar suite`.

.. _tests grammar suite:

The DecL grammar suite, every line of ``test_suite.agg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``src/aggregate/agg/test_suite.agg`` is a long file of example DecL programs, organized into categories A to O: frequencies, severities, reinsurance, distortions, case studies, papers. ``tests/test_decl_parser.py`` turns each line into its own pair of parametrized tests:

* ``test_line_parses``, the line parses to a valid ``(kind, name, spec)`` triple, so the program is grammatically legal.
* ``test_spec_matches_snapshot``, the parsed ``spec`` matches a stored reference in ``tests/data/expected_specs.json``. This catches semantic drift: if a grammar change silently alters how a program is interpreted, the spec no longer matches and the test fails.

Adding a line to ``test_suite.agg`` therefore adds test coverage for it automatically. Two companion corpora work the same way: ``tests/test_decl_unparser.py`` checks that every program round-trips through ``format_program``, and ``tests/test_grammar_sync.py`` checks the grammar reference against ``decl.lark``.

Snapshot or golden-master tests, "does the answer still match the file?"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A snapshot, or golden master, test compares today's computed output against a previously saved known-good copy on disk. These guard the numbers, not just "does it run".

* ``tests/test_baseline.py``, the characterization baseline. It loads ``tests/baseline/data/manifest.json``, rebuilds a curated set of aggregates and portfolios at a pinned grid, with fixed ``log2`` and ``bs`` so the answer is reproducible, recomputes their frames (``density_df``, ``summary_df``, ``stats_df`` and the rest) and compares element by element against stored ``.parquet`` files. It runs every case before reporting, and collects all divergences into one summary, for example::

      Sym.Dice / density_df / p_total : max abs 3.4e-09 at row '7.0'

  This is the project's main "did I move a number I did not mean to?" guard.
* ``tests/test_distortion_snapshot.py`` and ``tests/test_severity_layer_golden.py``, smaller golden-master checks for distortions and layered-severity moments.

The reference data is produced by capture scripts, ``tests/baseline/capture.py`` and the ``tests/capture_*.py`` files. See :ref:`tests regenerating`.

Feature suites, "this whole capability behaves correctly"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Larger files that exercise one feature end to end, usually mixing closed-form checks, identities, and regressions:

* ``tests/test_negative_x.py``, signed (profit and loss) severity and the output window. It includes exact closed-form checks, since a fixed sum of :math:`\{-2, 5\}` has known support :math:`\{-6, 1, 8, 15\}`, plus symmetry checks, the bucket and window estimator, and ``sev_density_df``.
* ``tests/test_reins_reporting.py``, ``tests/test_reins_buckets.py`` and ``tests/test_reins_bivariate.py``, the reinsurance reporting frames, the rebucketing schemes, and the joint (ceded, net) bivariate law.
* ``tests/test_distortion_*``, building, calibrating, and applying risk distortions.
* ``tests/test_bounds.py``, ``tests/test_splice_suite.py`` and ``tests/test_underwriter.py``, pricing bounds, spliced severities, and the ``build()`` entry point.
* ``tests/test_fcc_surface.py``, the executable half of the ``info`` layout described in :ref:`info string`.
* ``tests/test_library_recipes.py``, which runs the Solution and Check of every documented library entry, so the library's own stated invariants are part of the suite. See :ref:`uw recipes`.

Error and robustness tests, "does it fail gracefully?"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_parser_errors.py`` and ``tests/test_parser_errors_integration.py`` feed deliberately broken DecL and assert that the error message is clear and points at the right spot. Good error messages are a feature, so they are tested too.

Validation tests, "is the FFT answer trustworthy?"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_validation.py`` exercises the library's own self-check, ``valid`` and ``validation_explanation``, which flags when the discretization grid is too coarse (aliasing) or moments do not reconcile. These tests confirm the self-check fires when it should and stays quiet when the model is fine. The machinery is described in :ref:`agg pipeline validation`.

Fixtures and ``conftest.py``
----------------------------

A fixture is reusable setup shared by many tests, so each test does not repeat it. Fixtures live in ``tests/conftest.py``, a special file pytest loads automatically, and are requested by naming them as a function argument:

.. code-block:: python

   def test_something(underwriter):     # 'underwriter' is a fixture
       ...                              # pytest builds it and passes it in

``aggregate``'s ``conftest.py`` provides two: ``test_suite_lines``, all the preprocessed DecL lines from ``test_suite.agg``, and ``underwriter``, an :class:`Underwriter` with those lines pre-loaded. Both are ``scope="session"``, so they are built once and reused across the whole run, which is fast.

Keeping ``decl-testers.agg`` in sync
------------------------------------

When a hand-written test such as ``test_negative_x.py`` uses DecL programs, the same programs are mirrored into ``src/aggregate/agg/decl-testers.agg`` under a matching section. That file is a human-readable catalogue of the programs the tests rely on. It is documentation rather than something auto-run, but it keeps the language examples discoverable and in one place. Add to it whenever you add DecL-driven tests.

.. _tests regenerating:

Regenerating reference data
---------------------------

Snapshot and golden tests compare against saved files. When you make a change that is supposed to move the numbers, or move a column, you regenerate the reference deliberately, never automatically:

.. code-block:: bash

   uv run python tests/baseline/capture.py          # the characterization baseline
   uv run python tests/capture_severity_golden.py   # severity golden master
   uv run python tests/capture_distortion_snapshot.py

Then commit the regenerated ``.parquet`` and ``.json`` in the same commit as the code change, with a message explaining why the numbers moved. The discipline is: if a snapshot test fails, first decide whether the change was intended. If it was, regenerate and explain. If it was not, you just caught a bug, so fix the code rather than regenerating.

.. warning::

   Regenerating bakes in whatever the code currently produces. Only do it once you have confirmed the only differences are the ones you intended. The baseline failure report lists each divergence, which makes that easy to check.

Reading a failure
-----------------

A typical failure looks like::

   FAILED tests/test_negative_x.py::test_fixed_n_closed_form -
       assert {-6: 0.125, 1: 0.375, ...} == {-6: 0.125, 1: 0.374, ...}

pytest shows the assert that failed and both sides of the comparison. The workflow is:

1. read which test and which assertion;
2. reproduce just that test, ``uv run pytest -k fixed_n_closed_form -l``;
3. decide whether it is a bug in the code, or an intended change that needs the test or snapshot updated.

Tolerances matter. FFT results are exact only up to floating-point rounding, so numeric tests compare with a small tolerance, using ``abs(x - y) < 1e-12``, ``np.allclose``, or the project's noise-aware helpers, rather than ``==``.

Quick reference
---------------

=========================================  ==============================================
Command                                    Does
=========================================  ==============================================
``uv run pytest``                          Run the fast suite (``slow`` deselected)
``uv run pytest -m 'slow or not slow'``    Run absolutely everything
``uv run pytest -m slow``                  Run only the quarantined heavy cases
``uv run pytest -q``                       Quietly: dots and a summary
``uv run pytest -k NAME``                  Run tests whose name contains ``NAME``
``uv run pytest FILE``                     Run one file
``uv run pytest FILE::TEST``               Run one exact test
``uv run pytest -x``                       Stop at the first failure
``uv run pytest -l``                       Show local variables on failure
``uv run pytest -v``                       List every test name
``uv run pytest -n0``                      One process, for a breakpoint or fixed order
``uv run pytest --durations=20``           Report the twenty slowest tests
=========================================  ==============================================

Add a test whenever you add a feature or fix a bug, choosing the test that would have caught it. Run the relevant file as you work, and run the whole suite before you finish.
