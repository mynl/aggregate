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
    assert plt.rcParams["axes.facecolor"] == "lightsteelblue"
    assert plt.rcParams["figure.facecolor"] == "aliceblue"


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
    assert inside == "lightsteelblue"
    assert before == after  # restored on exit


def test_context_overrides():
    plt.rcdefaults()
    with aggregate.style.context(**{"figure.figsize": (5.5, 3.5)}):
        assert tuple(plt.rcParams["figure.figsize"]) == (5.5, 3.5)
        # base style still applies for non-overridden keys
        assert plt.rcParams["axes.facecolor"] == "lightsteelblue"
