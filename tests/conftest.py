"""Shared pytest fixtures for the aggregate test suite."""

from pathlib import Path

import pytest

from aggregate.constants import reset_warn_once
from aggregate.parser import UnderwritingLexer
from aggregate.underwriter import Underwriter

TEST_SUITE_PATH = Path(__file__).parent.parent / "src" / "aggregate" / "agg" / "_test_suite.agg"


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """Re-arm the once-per-session warnings before every test.

    ``warn_once`` state is process-wide by design, which is right for a
    session and wrong for a test suite: without this, the first test to
    provoke a condition silences ``pytest.warns`` for every later test in the
    same worker, and which test that is depends on ordering and on the xdist
    split. Reset before, not after, so a test can inspect the registry it
    leaves behind.
    """
    reset_warn_once()


@pytest.fixture(scope="session")
def test_suite_lines() -> list[str]:
    """All preprocessed DecL lines from aggregate/agg/_test_suite.agg."""
    text = TEST_SUITE_PATH.read_text(encoding="utf-8")
    return UnderwritingLexer.preprocess(text)


@pytest.fixture(scope="session")
def underwriter(test_suite_lines):
    """An Underwriter with the _test_suite.agg recipes tolerantly preloaded.

    Each line is parsed and added to the recipe base; parse failures are
    swallowed here so they surface as individual test failures in the
    parametrized parse tests rather than as a fixture error.
    """
    uw = Underwriter()
    for line in test_suite_lines:
        try:
            kind, name, spec = uw.parser.parse(uw.lexer.tokenize(line))
        except Exception:
            continue
        uw.add_recipe(kind, name, spec, line)
    return uw


# ---------------------------------------------------------------------------
# The 'matrix' ChartDoc, shared by test_chartdoc_matrix (structure) and
# test_chartdoc_render (the approved picture).
#
# It lives here rather than in either test module because a cross test module
# import does not resolve under xdist workers, and because the two files must
# gate the *same* document: if they drift, the baseline stops meaning what the
# structure tests assert. No library object emits a 'matrix' panel (the kind
# exists for plugins), so there is no build(...) that produces one.
# ---------------------------------------------------------------------------

def matrix_data():
    """Two positions against two readings, read in opposite directions."""
    from aggregate.charts import MatrixData
    return MatrixData(
        rows=('gross book', 'QS'),
        columns=('gini ph', 'margin'),
        values=((1.0, 1.0), (1.60, 0.40)),
        annotations=(('0.209', '7.3%'), ('0.314', '2.9%')),
        center=1.0, neutral=0.05,
        # The book row is read the other way: pricing above the reference is an
        # improvement for it, and paying above the reference is not for the QS.
        row_polarity=(-1, 1),
        row_groups=('book', 'aggregate'),
        column_groups=('family', 'point'),
    )


def matrix_document(matrix=None):
    """The one panel ``matrix`` ChartDoc."""
    from aggregate.charts import ChartAxis, ChartDoc, ChartSeries, Panel
    matrix = matrix_data() if matrix is None else matrix
    return ChartDoc(
        name='relativity', title='Relativity',
        axes=(ChartAxis(id='x', label='reading', kind='category'),
              ChartAxis(id='y', label='position', kind='category'),
              ChartAxis(id='z', label='multiple of gross')),
        panels=(Panel(id='m', kind='matrix', x_axis='x', y_axis='y',
                      z_axis='z'),),
        series=(ChartSeries(name='relativity', role='identity', panel_id='m',
                            matrix=matrix),))


# The fixtures are named for what they are, not 'doc' / 'matrix', which are far
# too generic for the global fixture namespace. Test modules wanting the short
# names wrap these locally.

@pytest.fixture
def matrix_chartdoc_data():
    return matrix_data()


@pytest.fixture
def matrix_chartdoc(matrix_chartdoc_data):
    return matrix_document(matrix_chartdoc_data)
