"""Regression tests for the hygiene-3 batch (1.0.0a52).

Covers three independent nits:

- **Item 1** -- Python-style ``_`` digit-group separators in DecL numbers
  (``10_000_000``), with leading / trailing / doubled underscores rejected.
- **Item 3** -- ``Aggregate._sev_label`` no longer raises on an array-like
  ``self.sevs`` (multi-component severity).
- **Item 4** -- ``of`` accepted as a share synonym for ``so`` / ``po`` in a
  reinsurance clause.

These are grammar / control-flow nits: no movement of the numeric baseline.
"""

import numpy as np
import pytest

from aggregate import build
from aggregate.underwriter import Underwriter


@pytest.fixture(scope="module")
def uw():
    return Underwriter()


def _parse(uw, program):
    return uw.parser.parse(uw.lexer.tokenize(program))


# ----------------------------------------------------------------------
# Item 1 -- underscore digit separators
# ----------------------------------------------------------------------

@pytest.mark.parametrize("withsep,plain", [
    ("10_000_000", "10000000"),
    ("1_000.5", "1000.5"),
    ("1_0e3", "10e3"),
])
def test_underscore_separators_equal_plain(uw, withsep, plain):
    """A number with ``_`` separators parses to the same value as without."""
    a = _parse(uw, f"agg U {withsep} claims dsev [1] poisson")
    b = _parse(uw, f"agg U {plain} claims dsev [1] poisson")
    assert a[2]["exp_en"] == b[2]["exp_en"]


@pytest.mark.parametrize("bad", ["_1", "1_", "1__0"])
def test_underscore_malformed_rejected(uw, bad):
    """Leading / trailing / doubled underscores are not valid numbers."""
    with pytest.raises(Exception):
        _parse(uw, f"agg U {bad} claims dsev [1] poisson")


# ----------------------------------------------------------------------
# Item 3 -- array-ambiguous truth test in _sev_label
# ----------------------------------------------------------------------

def test_sev_label_multicomponent_no_raise():
    """A multi-component severity makes ``self.sevs`` an ndarray; the old
    ``if not self.sevs:`` raised 'truth value of an array ... ambiguous'."""
    a = build("agg Mix 10 claims "
              "sev lognorm [50 100 200] cv [1 1.5 2] wts [.4 .3 .3] poisson")
    assert isinstance(a.sevs, np.ndarray) and len(a.sevs) > 1
    assert a._sev_label() == "3 components"
    # exercises _sev_label via the tail-text path too
    assert isinstance(a.tail_description, str) and a.tail_description


# ----------------------------------------------------------------------
# Item 4 -- the two readings of a placement quantity.
#
# This item shipped as the ``of`` share synonym. ``of`` and ``so`` were retired
# at 1.0.0a249 ([Single-Placement-Keyword]), leaving ``po`` as the one spelling;
# what survives the retirement is the percent-versus-bare distinction, which is
# what these now pin. See dev/done/plan-single-placement-keyword.md.
# ----------------------------------------------------------------------

_PLACEMENT = ("agg B 1 claim 10000 xs 0 sev lognorm 120 cv 1.5 "
              "occurrence net of {qty} {kw} 6000 xs 4000 poisson")


def test_po_percent_is_the_share(uw):
    """``90% po`` uses the percentage as the share directly."""
    spec = _parse(uw, _PLACEMENT.format(qty="90%", kw="po"))[2]["occ_reins"]
    assert spec == [(0.9, 6000.0, 4000.0)]


def test_po_bare_amount_is_amount_over_limit(uw):
    """A bare amount with ``po`` reads as ``amount / limit``."""
    spec = _parse(uw, _PLACEMENT.format(qty="3000", kw="po"))[2]["occ_reins"]
    assert spec == [(0.5, 6000.0, 4000.0)]


@pytest.mark.parametrize("kw", ["so", "of"])
def test_retired_placement_keywords_are_rejected(uw, kw):
    """``so`` and ``of`` no longer start a placement clause."""
    with pytest.raises(ValueError):
        _parse(uw, _PLACEMENT.format(qty="90%", kw=kw))


def test_so_is_an_ordinary_identifier_again(uw):
    """Retiring ``SHARE_OF`` returned ``so`` to the identifier space."""
    assert _parse(uw, "agg so 3 claims dsev [1 2 3] poisson")[:2] == ("agg", "so")
