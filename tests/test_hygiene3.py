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
# Item 4 -- 'of' as a share synonym
# ----------------------------------------------------------------------

def test_of_equals_so_and_po(uw):
    """``90% of`` yields the same (share, limit, attach) as ``so`` / ``po``."""
    base = ("agg B 1 claim 10000 xs 0 sev lognorm 120 cv 1.5 "
            "occurrence net of 90% {kw} 6000 xs 4000 poisson")
    of_ = _parse(uw, base.format(kw="of"))[2]["occ_reins"]
    so_ = _parse(uw, base.format(kw="so"))[2]["occ_reins"]
    po_ = _parse(uw, base.format(kw="po"))[2]["occ_reins"]
    assert of_ == so_ == po_ == [(0.9, 6000.0, 4000.0)]


def test_of_bare_amount_is_share(uw):
    """A bare amount with ``of`` reads as ``amount / limit`` (share), like ``so``."""
    of_ = _parse(uw, "agg B 1 claim 10000 xs 0 sev lognorm 120 cv 1.5 "
                     "occurrence net of 3000 of 6000 xs 4000 poisson")[2]["occ_reins"]
    so_ = _parse(uw, "agg B 1 claim 10000 xs 0 sev lognorm 120 cv 1.5 "
                     "occurrence net of 3000 so 6000 xs 4000 poisson")[2]["occ_reins"]
    assert of_ == so_ == [(0.5, 6000.0, 4000.0)]
