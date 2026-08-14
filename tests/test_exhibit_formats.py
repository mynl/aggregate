"""Tests for the format sheets (``dev/plan-formats.md``, ``[Format-Sheets]``).

Two things are under test and they are different in kind. The **loader** is
ordinary machinery: parse, merge nearest wins across the three stop search
path, resolve style references, validate once at load. The **sheets** are
content, and the tests over them assert the two invariants that keep the pair
honest: every shipped entry is a format greater_tables understands, and the
insurer sheet holds only entries that actually differ from the raw one. An
overlay row that agrees with its base is not a statement, it is a copy waiting
to go stale, which is the whole argument for a delta over a second full sheet.
"""
from __future__ import annotations

import pytest

from aggregate.exhibits import _formats
from aggregate.exhibits._formats import (
    SHEET_FILENAMES, FormatSheet, format_sheet, reload_format_sheets,
    sheet_paths,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """No test inherits another's cached sheet, or its working directory."""
    reload_format_sheets()
    yield
    reload_format_sheets()


# --- the shipped sheets ------------------------------------------------------

def test_the_shipped_sheets_are_found_and_parse():
    for kind in SHEET_FILENAMES:
        paths = sheet_paths(kind)
        assert len(paths) == 3, 'package, user dir, working directory'
        assert paths[0].is_file(), f'{kind} sheet missing from package data'
    sheet = format_sheet('raw')
    assert isinstance(sheet, FormatSheet)
    assert sheet.sources[0] == sheet_paths('raw')[0]
    assert len(sheet.columns) > 40


def test_raw_readings():
    """Spot checks over the vocabulary, one per style and one literal."""
    columns = format_sheet('raw').columns
    assert columns['CV'] == '.1%'            # style ratio
    assert columns['L'] == ',.7g'            # style money
    assert columns['p'] == '.5f'             # style probability
    assert columns['clipped'] == '.2e'       # style residual
    assert columns['Skew'] == '.3g'          # a literal
    assert columns['VaR/Mean'] == '.3f'


def test_the_insurer_sheet_is_a_delta():
    """Absent means same, and present means genuinely different."""
    raw = format_sheet('raw')
    insurer = format_sheet('insurer')
    assert set(insurer.columns) == set(raw.columns), \
        'the overlay adds no columns of its own today'
    differ = {label for label in raw.columns
              if raw.columns[label] != insurer.columns[label]}
    # the g to f swap, plus every column pointing at the redefined money style
    assert 'Skew' in differ and 'error' in differ and 'L' in differ
    assert 'CV' not in differ and 'LR' not in differ and 'T' not in differ
    declared = _formats._declaration('insurer')[0]
    for label in declared['columns']:
        assert label in differ, \
            f'insurer entry {label!r} agrees with the raw sheet; delete it'


def test_redefining_a_style_moves_every_column_that_points_at_it():
    """One line in the overlay carries the whole money vocabulary."""
    insurer = format_sheet('insurer').columns
    for label in ('L', 'M', 'P', 'Q', 'a', 'E', 'C', 'VaR', 'TVaR', 'xsVaR'):
        assert insurer[label] == ',.2f', label
    assert format_sheet('raw').columns['VaR'] == ',.7g'


def test_a_tag_style_stamps_the_column_tag():
    """``ratio`` is greater_tables' own tag, so it travels as one."""
    sheet = format_sheet('raw')
    assert sheet.tags['CV'] == 'ratio'
    assert sheet.tags['LR'] == 'ratio'
    assert 'PQ' not in sheet.tags, 'a ratio by nature, but read as a multiple'
    _formatters, selectors = sheet.block()
    assert set(selectors) == {'ratio_cols'}
    assert {'CV', 'LR', 'ROE', 'coc', 'Premium spent'} <= set(
        selectors['ratio_cols'])


def test_the_scoped_section_wins_for_one_exhibit():
    """The `P` collision: premium everywhere, the probability ladder on tail."""
    sheet = format_sheet('raw')
    assert sheet.block()[0]['P'] == ',.7g'
    assert sheet.block('tail')[0]['P'] == '.5f'
    assert sheet.block('summary')[0]['P'] == ',.7g'
    # and under insurer, where the money reading would collapse the ladder
    assert format_sheet('insurer').block('tail')[0]['P'] == '.5f'


def test_block_translates_keys_through_a_renamer():
    """greater_tables keys on the displayed label, so the sheet must follow."""
    sheet = format_sheet('raw')
    rename = {'CV': 'Coefficient of variation'}.get
    formatters, selectors = sheet.block(rename=lambda c: rename(c) or c)
    assert 'CV' not in formatters
    assert formatters['Coefficient of variation'] == '.1%'
    assert 'Coefficient of variation' in selectors['ratio_cols']


# --- the loader --------------------------------------------------------------

def test_sheets_are_cached_per_perspective():
    assert format_sheet('raw') is format_sheet('raw')
    assert format_sheet('raw') is not format_sheet('insurer')
    first = format_sheet('raw')
    reload_format_sheets()
    assert format_sheet('raw') is not first


def test_perspective_accepts_the_enum_and_a_string():
    from aggregate.exhibits import Perspective
    assert format_sheet(Perspective.INSURER) is format_sheet('insurer')
    assert format_sheet('INSURER') is format_sheet('insurer')
    # an unimplemented perspective has no overlay, so it reads the raw sheet
    assert format_sheet('insured').columns == format_sheet('raw').columns


def test_the_working_directory_wins_and_inherits_the_rest(tmp_path, monkeypatch):
    """A one line sheet changes one reading, per the `.agg` override rule."""
    (tmp_path / 'formats-raw.yaml').write_text(
        "columns:\n  CV: '.3%'\n", encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    sheet = format_sheet('raw')
    assert sheet.columns['CV'] == '.3%'      # the near sheet wins
    assert sheet.columns['Skew'] == '.3g'    # everything else is inherited
    assert sheet.sources[-1] == tmp_path / 'formats-raw.yaml'


def test_a_local_style_redefinition_reaches_the_shipped_columns(
        tmp_path, monkeypatch):
    """Styles merge before columns resolve, which is what makes them useful."""
    (tmp_path / 'formats-raw.yaml').write_text(
        "styles:\n  money: 'si'\n", encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert format_sheet('raw').columns['P'] == 'si'


def test_a_local_sheet_may_scope_an_exhibit(tmp_path, monkeypatch):
    (tmp_path / 'formats-raw.yaml').write_text(
        "exhibits:\n  summary:\n    CV: '.4f'\n", encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    sheet = format_sheet('raw')
    assert sheet.block('summary')[0]['CV'] == '.4f'
    assert sheet.block('tail')[0]['CV'] == '.1%'
    # a scoped entry replaces the reading and the tag together
    assert 'CV' not in sheet.block('summary')[1]['ratio_cols']


def test_a_bad_format_names_the_file_and_the_key(tmp_path, monkeypatch):
    (tmp_path / 'formats-raw.yaml').write_text(
        "columns:\n  CV: 'nonsense'\n", encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"'CV'.*nonsense"):
        format_sheet('raw')
    with pytest.raises(ValueError, match='formats-raw.yaml'):
        format_sheet('raw')


def test_an_unknown_section_is_a_typo_not_a_silence(tmp_path, monkeypatch):
    (tmp_path / 'formats-raw.yaml').write_text(
        "colums:\n  CV: '.3%'\n", encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unknown section.*'colums'"):
        format_sheet('raw')


def test_a_sheet_that_is_not_a_mapping_says_so(tmp_path, monkeypatch):
    (tmp_path / 'formats-raw.yaml').write_text('- CV\n- Skew\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='a format sheet is a mapping'):
        format_sheet('raw')


def test_an_empty_sheet_contributes_nothing(tmp_path, monkeypatch):
    (tmp_path / 'formats-raw.yaml').write_text('# nothing here\n',
                                               encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    assert format_sheet('raw').columns['CV'] == '.1%'


def test_a_mapping_entry_reaches_the_fields_sugar_cannot(tmp_path, monkeypatch):
    """The FormatSpec mapping form: scale, prefix, suffix, paren negatives."""
    (tmp_path / 'formats-raw.yaml').write_text(
        "columns:\n"
        "  L:\n"
        "    kind: dec\n"
        "    digits: 1\n"
        "    scale: 0.001\n"
        "    suffix: 'k'\n"
        "    negative: paren\n", encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    entry = format_sheet('raw').columns['L']
    assert entry['suffix'] == 'k' and entry['negative'] == 'paren'
