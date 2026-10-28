***********************************
**Dec** Language  Reference
***********************************

.. To refresh the grammar listing below, run ``python -m aggregate.parser``,
   which regenerates 4_agg_language_reference/ref_include.rst from decl.lark
   (the single source of truth). The railroad diagram is regenerated separately
   (see hacks/decl_railroad.py).

.. updated 2026-08-13


This section describes how a DecL program is pre-processed, lexed, and parsed according to the grammar specification. It reports the results of interpreting the builtin test suite of programs.

The :doc:`Aggregate Overview <2_Aggregate_Overview>` describes DecL's design and purpose.

Help in Jupyter and the Loss Lab
================================

One page of DecL help ships with the library, in two forms. ``decl_help()`` returns the anatomy, "DecL at a glance": one ``agg`` statement taken apart slot by slot, a portfolio and a profit and loss built on it, and the alternative forms each slot accepts. ``decl_help('card')`` returns the card, "DecL on one page": a ten-box reference card that prints to one Letter landscape page. Both display in a Jupyter cell, and the aggregate Loss Lab opens them from the ``?`` in its program box and from its menu.

.. code-block:: python

    from aggregate import decl_help
    decl_help()          # the anatomy
    decl_help('card')    # the card

The return is a ``str`` holding an HTML fragment with its own scoped styles and no script, so a web page can serve it as is; ``DECL_HELP_FORMS`` lists the forms. The test suite builds every program in both pages, so the help follows the grammar.

Pre-Processing
==============

A program holds one or more statements. Two statements are separated either by
a **blank line** (a line that is empty or only whitespace — the markdown
paragraph model) or by a **semicolon at the end of a line** (the Python model,
so dense one-statement-per-line lists stay legal). Every other newline is just
whitespace, so a single statement may be laid out across as many lines, with
whatever indentation, as you like — a multi-line portfolio needs no trailing
markers, only that its units share one paragraph (no blank line between them).

Python and C++ style ``#`` / ``//`` comments are **transparent**: they never
separate statements. A full-line comment between the clause-lines of one
statement (e.g. a commented-out reinsurance clause) simply vanishes; the lines
around it stay in the same statement. The corollary is that a comment cannot
separate two statements — use a blank line or a ``;``.

Before passing to the lexer, the following pre-processing occurs.

1. Remove full-line ``#`` / ``//`` comments entirely, including their newline,
   so they leave no blank-line ghost
2. Strip trailing (inline) ``#`` / ``//`` comments through end of line
3. Remove ``\n`` inside ``[ ]`` (vectors) that appear from using ``f'{np.linspace(...)}'``
4. Turn a semicolon at the end of a line into a statement break
5. Split into statements on blank lines
6. Flatten each statement: its newlines and indentation collapse to spaces

.. note::

   The earlier ``\`` line-continuation has been removed. A statement now spans
   multiple lines for free, so a stray backslash is a lexer error rather than
   silently ignored.

Lexing
======

The following characters are ignored and may be used freely to improve readability: tab (remaining after pre-processing), colon, comma, and pipe.

Aggregate names must not include underscore. Portfolio names may include underscore. Names can include a period, ``A.Basic.01``.

Lexing is performed by Lark's dynamic, context-sensitive lexer directly from the
terminal definitions in the grammar (``aggregate/decl.lark``). Each terminal —
its regular expression, priority, and any keyword negative-lookahead — is part of
the grammar listing in :ref:`4_dec_Language_Reference:Dec Language Grammar Specification` below, so there
is no separate, hand-maintained token table. Keyword/ID disambiguation is handled
by terminal priorities (keywords priority 2, ``ID`` priority 1, builtin dotted
names priority 3), which is what makes the lexing contextual.

.. note::

   Earlier versions of this page reproduced a SLY ``tokens = {...}`` / ``ID['keyword'] = TOKEN``
   lexer table here. That is obsolete: the migration to Lark (Earley + dynamic
   lexer) in 2026 replaced the SLY remapping trick with terminal priorities, and
   the grammar file is now the single source of truth for both terminals and
   rules.

Numeric Expressions
===================

Every slot that takes a number accepts an arithmetic expression, evaluated at parse time: exposures, the informational premium, layers and attachments, frequency parameters, severity scales and shifts, reinsurance clauses, collars, and ranges.

The operator set splits in two. A **bare** expression, one written without surrounding parentheses, admits division ``/``, exponentiation ``**`` or ``^``, the exponential ``exp(...)``, and parentheses. Addition ``+``, subtraction ``-`` and multiplication ``*`` are legal **only inside parentheses**::

    agg TEST (4 + 3*2) claims (100_000/(1-.25)) premium 1000 xs 0
        sev (exp(-1 * .4**2/2)) * lognorm .4 poisson

Here ``(4 + 3*2)`` is a computed claim count of 10, ``(100_000/(1-.25))`` grosses a premium up for a 25% expense ratio, and ``(exp(-1 * .4**2/2))`` is the lognormal mean normalizer :math:`e^{-\sigma^2/2}` that makes the severity mean 1.

The parenthesis requirement is a design decision, not an oversight. Outside parentheses those three characters already mean something else: ``*`` is the severity scale operator (``10 * lognorm 10 cv .09``) and the portfolio homogeneous multiplier (``2 * agg.MyLine``), ``+`` is the severity shift and the portfolio sum, and ``-`` is the severity negation. Admitting them into bare expressions would make ``2 * 3 * agg.X`` genuinely ambiguous. Inside parentheses none of that applies, because a parenthesis holds exactly one expression.

Precedence runs, tightest first: parentheses, then ``exp``, then ``**`` and ``^``, then ``*`` and ``/``, then ``+`` and ``-``. Exponentiation is right-associative, so ``(2**3**2)`` is 512; everything else is left-associative, so ``(8/4/2)`` is 1.

Three behaviors are worth knowing.

**A minus sign glued to a number is part of that number, and so binds tighter than** ``**``. Writing ``(-.4**2)`` gives +0.16, because ``-.4`` lexes as a single numeric token and the square of a negative is positive. Python reads ``-.4**2`` as -0.16, because there the unary minus binds looser than the exponent. To negate a computed value in DecL, write ``(-1 * x)`` or ``(0 - x)``; the mean normalizer above uses the first form for exactly this reason.

**Parentheses hold one expression, never a list.** ``(1 -2)`` is the subtraction -1. Brackets remain the list syntax and are unaffected: ``[1 -2]`` is still the two-element vector, and ``[1 - 2]`` is a parse error.

**A percentage literal keeps its percent reading only when used literally.** Arithmetic returns a plain number, so ``(50% + 1)`` is 1.5, and a computed value in the ``po`` (part of) placement position reads as an absolute amount rather than a share.

Expressions evaluate as the program parses, so the specification carries plain numbers and the formula is not retained. The canonical text that :func:`aggregate.decl_writer.spec_to_decl` and :func:`aggregate.decl_writer.format_program` produce therefore shows the evaluated literal: the program above decompiles with ``10 claims 133333.33333333334 premium``. This is the long-standing canonical, not verbatim, contract, the same one that collapses ``exp(.5)`` to its float.

Exact Compounds (nttagg)
========================

The ``nttagg`` statement declares a bounded discrete compound that is computed **exactly**, as rational numbers, alongside the ordinary float FFT pipeline. It is the computational answer to Wilson and Keich's accuracy program: for a bounded integer lattice the number-theoretic transform over prime finite fields, reconstructed with the Chinese remainder theorem, has no rounding error at all, so the float pipeline can be audited bucket by bucket::

    nttagg DiceOfDice dfreq [1:100] dsev [1:100]              note{exact dice of dice}
    nttagg Skewed     dfreq [0 1 2] [1/4 1/2 1/4] dsev [1 2 10] [0.3 0.5 0.2]

The body is exactly ``dfreq`` then ``dsev`` then the usual trailer: no exposures, layers, reinsurance, ``approximate``, ``!``, or ``picks``, so boundedness is guaranteed syntactically, and ``nttagg`` cannot appear inside a portfolio, pnl, or bivariate. Outcomes must be nonnegative integers (clear denominators and rescale the lattice yourself). Probabilities are **text exact**: a decimal literal ``0.3`` is exactly ``3/10``, scientific notation ``3e-2`` is exactly ``3/100``, an explicit ``a/b`` is the fraction itself, and an omitted probability list is the exact uniform ``1/m``. Parenthesized arithmetic (``+ - * /``, and ``**`` with a whole-number power) is evaluated exactly too, so ``dsev [1:3] [1/2 (1/2-1/(13*11*(19**3))) (1/(13*11*(19**3)))]`` is exactly ``[1/2 980835/1961674 1/980837]`` and ``(0.1+0.2)`` is exactly ``3/10``; as in ``agg``, ``+``, ``-`` and ``*`` need the parentheses. The list must sum to exactly 1; duplicates combine and sort as in ``dhistogram``.

``build`` returns an :class:`aggregate.ntt.AggregateNTT`, an ordinary :class:`~aggregate.Aggregate` whose float pipeline is driven bit-identically from the exact spec (``float(Fraction)`` is correctly rounded), plus ``density_df`` columns ``p_exact_float`` and ``p_exact_log10`` (the latter readable in the deep tail, where the former underflows to zero), the exact rationals on the ``p_exact`` property (or ``p_exact_at(loss)`` for named buckets, which is the cheap way to ask), the per-bucket ``accuracy_df`` comparison, an ``ntt_audit`` record, and a ``stats_df`` ``exact`` column that validation measures against. Guardrails refuse monster cases up front with a size estimate (override with ``build(..., ntt_override=True)``); a case whose support needs more than the default ``log2`` cap asks for an explicit ``log2``; the 32-bit prime supply caps the envelope near a transform length of :math:`2^{22}`, intrinsic to the method. The transform phase is threaded across primes, ``min(4, os.cpu_count())`` by default and set with ``build(..., ntt_workers=n)`` (``1`` forces serial). Both heavy phases, the per-prime transform and the Chinese remainder reconstruction, run in the optional compiled ``aggregate-ntt`` package when that happens to be installed, which needs no attention: it is discovered rather than declared, the answers are identical either way, and ``build(..., ntt_engine='python')`` forces the pure-Python path if one is ever in dispute. The engine also lifts the prime-supply cap, since it carries 63-bit primes where the pure-Python transform is limited to 32-bit, so it needs about half as many and reaches cases the library alone refuses. ``ntt_audit`` names the engine that produced a record, and the ``aggregate.ntt`` logger reports the route at ``INFO``. See :mod:`aggregate.ntt` for the engine and the mathematics.

.. _franchise deductible:

Deductible Forms (franchise, disappearing)
==========================================

A layers clause ``y xs a`` applies an ordinary deductible: the insured retains the first ``a`` of every loss and the layer pays the excess, up to the limit ``y``. Two modifiers change only the retention. Under ``franchise``, once a loss exceeds the attachment the insured keeps nothing, and the whole loss is paid, subject to the limit. Under ``to d2 disappearing`` the retention shrinks as the loss grows and vanishes at a loss of ``d2``::

    agg Ordinary     100 claims 1000 xs 50                    sev lognorm 100 cv 2 poisson
    agg Franchise    100 claims 1000 xs 50 franchise          sev lognorm 100 cv 2 poisson
    agg Disappearing 100 claims 1000 xs 50 to 200 disappearing sev lognorm 100 cv 2 poisson
    agg Vector       100 claims [500 1000] xs [25 50] to [40 80] disappearing sev lognorm 100 cv 2 poisson

All three forms belong to one family of retentions. The retention on a loss :math:`x` is parameterized by the loss :math:`d_2` at which it reaches zero:

.. math::

    r(x) = \begin{cases}
        d_1 & x \le d_1 \\
        d_1 \dfrac{d_2 - x}{d_2 - d_1} & d_1 < x \le d_2 \\
        0 & x > d_2,
    \end{cases}

and the layer ``y xs d1`` pays :math:`g(x) = \min(x - r(x), y)`. Setting :math:`d_2 = \infty` gives the ordinary deductible, payment :math:`\min((x - d_1)^+, y)`. Setting :math:`d_2 = d_1` gives the franchise, payment :math:`\min(x, y)` for :math:`x > d_1` and zero otherwise, so no payment ever lies strictly between zero and :math:`\min(d_1, y)`. A value :math:`d_2 > d_1` is the disappearing deductible, written ``y xs d1 to d2 disappearing``: on :math:`(d_1, d_2]` it pays :math:`k (x - d_1)` with :math:`k = d_2 / (d_2 - d_1)`, rising from zero to the full loss :math:`d_2`, and above :math:`d_2` the whole loss, each capped at :math:`y`. The only constraint is :math:`d_1 \le d_2`; ``to d1 disappearing`` is the franchise and ``to inf disappearing`` the ordinary deductible.

**Limit convention.** The deductible applies first and the limit second, following the IRMI definition of a franchise deductible ("once it is met, the entire amount of the loss is paid, subject to the policy limit") and the first edition of *Loss Models*. The limit :math:`y` is the largest payment in every form: ``1000 xs 50 franchise`` pays :math:`\min(X, 1000)` once :math:`X > 50`. The payment reaches the limit at the ground-up loss :math:`d_1 + y` for the ordinary deductible, :math:`\max(d_1, y)` for the franchise, and for the disappearing deductible :math:`d_1 + y / k` when :math:`y \le d_2`, otherwise :math:`y`; this is the layer's ``detachment``. With :math:`y > d_1` the franchise pays :math:`d_1` plus the ordinary layer ``(y - d1) xs d1`` on every claim above :math:`d_1`; with :math:`y \le d_1` every such claim pays exactly :math:`y`.

**Conditional and unconditional.** As with an ordinary layer, the claim count is the count of losses exceeding the attachment, and the severity is conditional on :math:`X > d_1`. For the franchise its distribution function is :math:`(F(t) - F(d_1)) / S(d_1)` for payments :math:`d_1 < t < y`, zero below, and one at the limit, which carries the mass :math:`S(\max(d_1, y)) / S(d_1)`. For the disappearing deductible it is :math:`(F(d_1 + t / k) - F(d_1)) / S(d_1)` for payments :math:`t \le d_2` and :math:`(F(t) - F(d_1)) / S(d_1)` above, again up to the limit, where the mass is :math:`S(\text{detachment}) / S(d_1)`. The trailing ``!`` makes the severity unconditional in the usual way: the count is ground up and the probability :math:`F(d_1)` of a loss at or below the attachment sits at a payment of zero.

**Vectors.** ``franchise`` applies to every layer of a vector clause ``[...] xs [...]``, each with its own attachment. For ``disappearing``, ``to`` takes either one value for every layer or a vector ``to [...]`` with one retention end per layer, counting layers after the limits and attachments broadcast (``[500 1000] xs 50 to [60 70] disappearing`` is two layers); a vector of any other length is an error.

**Moments** come from the same partial expectations as an ordinary layer, so the theoretical moments that validation compares against are exact. The library entries ``SevLayer.Franchise`` and ``SevLayer.Disappearing`` set both forms beside ``SevLayer``.

.. _loss tables:

Event and Year Loss Tables (elt, yelt)
=====================================

An aggregate can be declared directly from a catastrophe model's event loss table (ELT) or from a year loss table (YELT) of simulated annual outcomes. The table replaces the exposure, severity and frequency clauses; occurrence and aggregate reinsurance, orientation and the trailer follow as usual, and the result is an ordinary :class:`~aggregate.Aggregate`::

    agg Basic as "My easy example" elt [[1 .1 100] [2 .1 40] [3 .1 80]]
    agg Shared elt unit Personal as "Personal lines" [[1 .1 300] [2 .1 100]]
               elt unit Commercial [[1 .1 200] [3 .1 200]]
               occurrence net of 1000 xs 500 aggregate net of 2000 xs 1000
    agg Annual as "Four equally likely years" yelt [0 0 100 300]
    agg Weighted yelt [[1 0 .5] [2 100 .3] [3 300 .2]] aggregate net of 150 xs 50

**Event loss tables.** Each row is ``[id rate mean]`` (ELT3), ``[id rate mean max]`` (ELT4) or ``[id rate mean max sdi sdc]`` (ELT6): an event identifier, its annual rate, the mean loss given the event, the maximum loss, and the independent and correlated standard deviations of the secondary uncertainty. Event identifiers are exact: an unsigned integer or a quoted string, so ``7`` and ``"7"`` name the same event and ``"007"`` another. An identifier names one physical event, so rows sharing it are the same occurrence. Let event :math:`e` have rate :math:`\lambda_e` and unit means :math:`\mu_{eu}`. Its occurrence loss is :math:`m_e = \sum_u \mu_{eu}`, and with :math:`\Lambda = \sum_e \lambda_e` the annual loss is compound Poisson with expected count :math:`\Lambda` and severity atoms :math:`m_e` of probability :math:`\lambda_e / \Lambda`, equal losses combined. The annual cumulants are :math:`\kappa_r = \sum_e \lambda_e m_e^r`, so ``Basic`` has rate 0.3, mean 22 and variance 1,800. A zero-loss event keeps its rate, and the probability of no annual loss is :math:`\exp[-\Lambda(1 - p_0)]` where :math:`p_0` is the severity's mass at zero.

**Units.** ``unit NAME`` names the business unit, peril or region a source belongs to, with an optional ``as`` label; an omitted unit is ``Total``. Several sources may contribute to one aggregate. Rows for the same event in different units are one occurrence: their means add before an occurrence treaty applies, and the event is counted once. Every row of an event carries the same rate, and an event appears at most once per unit. Units that share no events combine as independent catalogs; distinct catalogs must use distinct event identifiers.

**Maximums.** The ELT4 maximum is reported (the largest event exposure, the sum of an event's unit maximums, is in the audit) but does not change the law: ``[1 .1 100]`` and ``[1 .1 100 1000]`` give the same aggregate. ELT3 and ELT4 sources may be combined, and the combination is ELT3, dropping every maximum.

**Secondary uncertainty.** An ELT6 row ``[id rate mean max sdi sdc]`` makes the loss given the event random: a beta on :math:`[0, M]` with mean :math:`\mu` and standard deviation :math:`\sigma_I + \sigma_C`. When that variance is inadmissible, :math:`\sigma^2 \ge \mu(M - \mu)`, the row becomes the two-point law with mass :math:`\mu / M` at :math:`M` and the rest at zero, which keeps the mean; a row with both SDs zero is an atom at its mean. The annual loss is still compound Poisson with rate :math:`\Lambda`, now with severity the rate-weighted mixture of the event laws, so the annual cumulants are :math:`\kappa_r = \sum_e \lambda_e \mathsf P X_e^r`. For example ``agg Beta elt [[1 .1 50 100 10 10]]`` is a beta(2.625, 2.625) on :math:`[0, 100]` with annual mean 5 and variance 290. The model moments are exact and available with ``build(..., update=False)``. On a grid, the event laws are quantized by the `eltq <https://pypi.org/project/eltq/>`_ package, nearest node, with the mass above the top bucket kept on the top node. Unless ``bs`` is given, the bucket is chosen as for any aggregate and then grown, if need be, until the severity grid reaches far enough into the tail: the annual rate of events above the grid edge at most :math:`10^{-12}` and the part of each of the first three annual cumulants lying above the top node at most ``validation_eps / 10`` of it. A given ``bs`` is honored with a warning when it falls short. ``loss_table_summary_df`` and ``loss_table_validation_df`` report the measured tail. ``elt no2u ...`` converts an ELT6 source to ELT4, each row an atom at its mean, and mixing an ELT6 source without ``no2u`` with ELT3 or ELT4 sources is an error naming the sources that need it.

**Shared events.** When one event has rows for two or more units, the rows are rolled up into one row before modelling, the convention of Homer and Li and of Mitchell-Wallace et al.: means, maximums and correlated SDs add, and independent SDs add in quadrature, :math:`\sigma_{I,R} = (\sum_u \sigma_{I,u}^2)^{1/2}`. The rolled row is then classified and quantized like any one-row event, so the table updates normally and ``info`` shows ``shared events rolled up``. The roll-up is exact in the mean and approximate in the variance and tail of an event total. Units whose events are all distinct (separate wind and quake catalogs, say) are not shared.

**Installing eltq.** Secondary uncertainty needs eltq 2.13 or later, the optional ``elt`` extra: ``pip install aggregate[elt]``, or ``aggregate[all]``. Without eltq an ELT6 source without ``no2u`` raises ``ImportError``; everything else in this section works without it.

**Year loss tables.** A source is a flat vector of annual losses (YELT1), rows ``[year loss]`` (YELT2), or rows ``[year loss prob]`` (YELT3). Each row is a complete annual outcome, so the aggregate is ``dfreq [1] dsev`` over the annual totals, with no Poisson compounding: ``yelt [0 0 100 300]`` is ``dfreq [1] dsev [0 0 100 300]``, probability one half at zero, mean 100 and variance 15,000 (population moments). Years are equally likely unless probabilities are supplied, and supplied probabilities must sum to one. Zero-loss years must be listed: they determine the probability of no loss. Units are aligned by year identifier, or by position for flat vectors, and summed within each year before equal totals are combined, which keeps whatever dependence the simulation produced. Every unit supplies the same complete set of years. A YELT takes aggregate reinsurance only: an annual total does not identify its occurrences, so occurrence terms are refused, including through a reference ``agg NEW agg.YELT occurrence ...`` or a ``pnl`` engine.

**Files.** A source can be a quoted path instead of an inline table: ``elt unit Wind "wind.csv"`` or ``yelt "years.parquet"``. CSV, TSV, Parquet and Feather files are read (the columnar formats need ``pyarrow``). Columns are found by name after lowercasing and removing spaces and underscores, so ``Event ID`` is ``event_id``. For an ELT the names are ``id``, ``event_id``, ``event`` or ``eventid``; ``rate``, ``frequency``, ``freq`` or ``annual_rate``; ``mean``, ``mean_loss`` or ``loss`` (the mean given the event, not the rate times it); ``max``, ``max_loss``, ``maximum_loss``, ``exposure`` or ``limit``; ``sdi``, ``sd_indep``, ``sdind`` or ``stddevi``; ``sdc``, ``sd_corr``, ``sdcor`` or ``stddevc``. For a YELT they are ``year_id``, ``year``, ``trial_id``, ``trial``, ``simulation_id``, ``sample_id`` or ``id``; ``loss``, ``annual_loss`` or ``annual_loss_amount``; ``prob``, ``probability`` or ``p``. Either may have a ``unit`` or ``unit_id`` column, which splits one long file into units; a ``unit`` clause must then agree with it. The columns present set the form, two columns with the same meaning are an error, and other columns are ignored and listed in the audit. Identifiers are read as text, an integer column becomes its decimal text, and a floating point identifier column is refused. A relative path is resolved against the directory of the ``.agg`` file that declares it, or the working directory when the statement is built, and keeps that meaning if the directory later changes; ``to_agg`` writes it as an absolute path. The file is read when the aggregate is built, and the audit records the resolved path, size, modification time and SHA-256 digest of what was read. Rebuilding rereads the file; an existing aggregate, and ``Aggregate(**a.spec)``, never do.

**Limits.** A loss table body cannot be combined with ``claims``, ``dfreq``, ``sev`` or ``dsev`` clauses, or with ``approximate``, and an ``elt`` and a ``yelt`` source cannot share an aggregate. The table is data, so ``@``, ``*``, ``+`` and ``-`` on a reference to it are refused. A loss table may be a portfolio unit or a ``pnl`` engine; it cannot be a ``bivariate`` component. The ``loss table`` row of ``info`` summarizes the form, event or year count and units. Three frames report the rest: ``loss_table_summary_df`` lists what the declaration read, combined and per source (with each file's resolved path and SHA-256 digest); ``loss_table_stats_df`` sets the severity and annual mean, CV and skewness computed from the table beside the model's and the discretized estimate; ``loss_table_validation_df`` lists each check with its value, tolerance and status. ``elt``, ``yelt``, ``no2u`` and ``unit`` are reserved words.

Dec Language Grammar Specification
===================================

Here is the full DecL Grammar and a `grammar railroad diagram <_static/diagram.xhtml>`_.

.. The grammar below is regenerated from decl.lark by ``python -m aggregate.parser``.

.. include:: 4_agg_language_reference/ref_include.rst

.. _test suite programs:

Test Suite Programs
===================

The parser corpus (``aggregate/agg/_test_suite.agg``) is exercised by the
pytest suite: each line of the file becomes its own parametrized test case, one
asserting it parses and one checking the resulting spec against a captured
snapshot. The snapshot is regenerated from the current Lark parser, so it is a
change detector rather than a correctness oracle. Run::

    uv run pytest

The full ``.agg`` source:

.. literalinclude:: ../src/aggregate/agg/_test_suite.agg
   :language: agg

To only parse the file from Python::

    from aggregate import build
    filename = build.default_dir / '_test_suite.agg'
    assert filename.exists()

    build.logger_level(30)
    df = build.interpret_file(filename=filename)

    df.query('error != 0')


Reading Parse Errors
=====================

When ``build()`` fails on a DecL typo, the wrapping :class:`ValueError`
carries a structured :class:`~aggregate.parser_errors.ErrorReport` on
its ``.report`` attribute. The report has 1-indexed line and column,
the source line (windowed to the caret on long inputs), a caret
marker, friendly "expected" labels, and ``difflib``-derived "did you
mean" suggestions.

Some mistakes produce a report that is accurate and still unhelpful,
because the offending token is fine in itself and the problem is the
grammar rule around it. For those, ``report.hint`` carries a sentence
naming the rule. Writing ``500 fixed expense as FE and 15% premium
expense as Comm`` reports ``Unexpected 'and'``, and the hint explains
that ``as`` closes an expense group, so the ``and`` has nothing to join:
drop it to leave two groups (one expense leg each), or move the ``as``
label after the last ``and``-joined term to combine them into one leg.
The hint is ``None`` when no rule matches, and it appears in both
``str(e)`` and ``render()``.

``str(e)`` is the one-line summary — ``DecL parse error at line L,
column C: Unexpected '...'. Did you mean: ...?`` — so the default
Python / Jupyter traceback footer is already useful without any
opt-in. ``e.report.render()`` returns the multi-line block with the
caret-annotated source line; the ``aggregate.underwriter`` logger
emits this automatically at ``ERROR`` level on every failed
``build()``. The three patterns below cover the common use-cases.

**1. Notebook / REPL — show the formatted error before the traceback.**

.. code-block:: python

    try:
        build('agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5')
    except ValueError as e:
        print(e.report.render())     # caret + "Did you mean..."
        raise                        # re-raise to keep the traceback

The rendered output looks like::

    DecL parse error at line 1, column 39:

      agg X 100 claims sev lognorm 100 cv 2 mixd poisson 0.5
                                            ^^^^

    Unexpected 'mixd'. Did you mean: mixed? Expected: 'mixed', 'occurrence', '/', '+', '-', ...

For long programs (longer than ~80 characters) the source line is
windowed around the caret with ``... `` / `` ...`` markers and
word-boundary snapping, so the caret stays on a single terminal row.

**2. Script that wants the suggestion programmatically.**

.. code-block:: python

    try:
        build(text)
    except ValueError as e:
        if getattr(e, 'report', None) and e.report.suggestions:
            print(f"Did you mean: {e.report.suggestions[0]}?")
        raise

The ``getattr`` guard makes the snippet robust against non-parse
``ValueError`` (e.g. semantic validation failures from the transformer)
which don't carry a ``.report``.

**3. IPython traceback hook — auto-format every parse error in a session.**

For users who want the report rendered automatically in every Jupyter
cell, install a one-shot traceback hook (typically in
``~/.ipython/profile_default/startup/decl_errors.py``):

.. code-block:: python

    from IPython import get_ipython

    def _showtb(self, etype, evalue, tb, **kw):
        report = getattr(evalue, 'report', None)
        if report is not None:
            print(report.render())
        return self._showtraceback_original(etype, evalue, tb, **kw)

    ip = get_ipython()
    ip._showtraceback_original = ip.showtraceback
    ip.showtraceback = _showtb.__get__(ip)

The library deliberately does **not** install this hook on import —
recipe (3) is shown here for power users who want it, and is easy to
undo.

The same report is also written via the ``aggregate.underwriter``
logger at ``ERROR`` level on every failed ``build()``, so a notebook
with logging configured will see the rendered text adjacent to the
traceback without any opt-in.


Parser Implementation
=======================

The parser is built using `Lark <https://lark-parser.readthedocs.io/>`_ with an Earley backend and a dynamic, context-sensitive lexer. The grammar lives in ``aggregate/decl.lark`` and is the single source of truth — the listing in :ref:`4_dec_Language_Reference:Dec Language Grammar Specification` is regenerated from it. Earley dissolves the shift/reduce conflicts the previous SLY (LALR) implementation needed to hand-tune; the dynamic lexer plus tightened ``ID`` rule keep the grammar unambiguous.
