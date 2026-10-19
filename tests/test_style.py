"""Smoke tests for :mod:`aggregate.style`."""

import matplotlib.pyplot as plt
import pandas as pd

import aggregate.style


def test_rc_params_returns_dict():
    params = aggregate.style.rc_params()
    assert isinstance(params, dict)
    assert "axes.facecolor" in params


def test_rc_params_returns_copy():
    a = aggregate.style.rc_params()
    a["axes.facecolor"] = "red"
    b = aggregate.style.rc_params()
    assert b["axes.facecolor"] != "red"


def test_use_sets_facecolor():
    aggregate.style.use(pandas=False)
    assert plt.rcParams["axes.facecolor"] == "white"
    assert plt.rcParams["figure.facecolor"] == "white"


def test_use_pandas_toggle():
    pd.reset_option("display.width")
    default = pd.get_option("display.width")
    aggregate.style.use(pandas=False)
    assert pd.get_option("display.width") == default
    aggregate.style.use(pandas=True)
    assert pd.get_option("display.width") == 120


def test_context_scopes_rcparams():
    plt.rcdefaults()
    before = plt.rcParams["axes.facecolor"]
    with aggregate.style.context():
        inside = plt.rcParams["axes.facecolor"]
    after = plt.rcParams["axes.facecolor"]
    assert inside == "white"
    assert before == after  # restored on exit


def test_context_overrides():
    plt.rcdefaults()
    with aggregate.style.context(**{"figure.figsize": (5.5, 3.5)}):
        assert tuple(plt.rcParams["figure.figsize"]) == (5.5, 3.5)
        # base style still applies for non-overridden keys
        assert plt.rcParams["axes.facecolor"] == "white"


def test_house_type_and_face_in_a_pdf(tmp_path, caplog):
    """``Aggregate.plot()`` under the style prints ticks and legend at 8pt,
    labels and titles at 9pt, the suptitle at matplotlib's ``large``, and
    embeds STIXGeneral alone, with no font fallback logged.

    ``dev/plan-library-figures.md`` (``[Library-Figures-House-Type]``). The
    program mirrors a line of ``src/aggregate/agg/decl-testers.agg``. Read
    from the uncompressed content stream: ``Tf`` sets each span's size.
    """
    import logging
    import re

    from aggregate import build

    a = build('agg P.Tail.LN 5 claims sev lognorm 100 cv 3 poisson')
    path = tmp_path / 'agg.pdf'
    with caplog.at_level(logging.WARNING, logger='matplotlib.font_manager'):
        with aggregate.style.context(**{'pdf.compression': 0}):
            fig = a.plot()
            fig.savefig(path)
            plt.close(fig)
    raw = path.read_bytes().decode('latin-1')
    sizes = {float(s) for s in re.findall(r'/F\d+ ([\d.]+) Tf', raw)}
    assert sizes == {8.0, 9.0, 10.8}
    faces = {re.sub(r'^[A-Z]{6}\+', '', f)
             for f in re.findall(r'/BaseFont /(\S+)', raw)}
    assert faces and all(f.startswith('STIXGeneral') for f in faces), faces
    assert not [r for r in caplog.records if 'findfont' in r.getMessage()]
