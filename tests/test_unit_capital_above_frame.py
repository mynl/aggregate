"""``unit_capital_at`` at and above the top of the augmented frame (#12).

Two behaviors, both on a small compound Poisson book whose augmented frame
stops at loss 64 while ``density_df`` carries support to 69:

* An asset level above the frame uses the last row with a warning, as
  :meth:`Portfolio.pricing_at` does, instead of raising a numpy broadcast
  error. :meth:`Portfolio.pentagon_at` now warns when it does the same.
* A warning is logged whenever the asset level reaches a layer with
  ``S <= VALIDATION_NOISE``, whichever units are requested. There the split of
  capital between units is not determined by the model, and before this the
  only alarm was the reconciliation check, which runs for all units only.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from aggregate import build
from aggregate.spectral import Distortion

WANG = Distortion('wang', 0.4)


@pytest.fixture(scope='module')
def port():
    """A small book with a long, thin Poisson tail."""
    return build('port UC.Frame agg A 2 claims dsev [1 2 4] poisson '
                 'agg C 1 claim dsev [5] fixed', log2=10, bs=1)


@pytest.fixture(scope='module')
def aug(port):
    return port.apply_distortion(WANG, allocation='linear')


def _warned(caplog, text):
    return any(text in r.getMessage() for r in caplog.records
               if r.levelno >= logging.WARNING)


def test_above_the_frame_uses_the_last_row(port, aug, caplog):
    """No broadcast error; the result is the last row's, with a warning."""
    top = float(aug.loss.iloc[-1])
    at_top = port._unit_capital_at(aug, top, WANG)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        above = port._unit_capital_at(aug, top + 5, WANG)
    assert _warned(caplog, 'above the augmented frame')
    assert above.keys() == at_top.keys()
    for unit in at_top:
        assert above[unit] == at_top[unit]


def test_pentagon_at_warns_when_it_uses_the_last_row(port, aug, caplog):
    """``pentagon_at`` matches ``pricing_at``: last row, and it says so."""
    top = float(aug.loss.iloc[-1])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        peg = port.pentagon_at(WANG, a=top + 100, unit='A', allocation='linear')
    assert _warned(caplog, 'pentagon_at: asset level')
    assert np.isfinite(peg.Q)


@pytest.mark.parametrize('units', [None, ['A'], ['C']])
def test_far_tail_alarm_for_any_units(port, aug, units, caplog):
    """The far tail warning fires whether one unit or all are requested."""
    top = float(aug.loss.iloc[-1])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port._unit_capital_at(aug, top, WANG, units=units)
    assert _warned(caplog, 'is not reliable there')


def test_far_tail_alarm_on_a_single_unit_pentagon(port, aug, caplog):
    """``pentagon_at(unit=...)`` in the far tail is no longer silent."""
    top = float(aug.loss.iloc[-1])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port.pentagon_at(WANG, a=top, unit='A', allocation='linear')
    assert _warned(caplog, 'is not reliable there')


@pytest.mark.parametrize('p', [0.99, 0.999, 0.99999])
def test_no_alarm_at_ordinary_levels(port, p, caplog):
    """Ordinary capital standards stay quiet."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        port.pricing_at(WANG, p=p, allocation='linear')
    assert not _warned(caplog, 'is not reliable there')
    assert not _warned(caplog, 'above the augmented frame')
