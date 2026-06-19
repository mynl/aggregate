"""DecL -> Distortion construction via the flat number-list grammar (1.0.0a27).

``distortion NAME kind n1 n2 ...`` maps the flat parameter list onto the kind's
natural keyword arguments through ``Distortion.decl_spec`` (driven by each
subclass's ``decl_params``); the parser holds no per-kind knowledge and the old
``_distortion_spec`` translation table is gone.

The DecL programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` under the
W. distortion section.
"""

import pytest

from aggregate import build
from aggregate.spectral import Distortion


# ----------------------------------------------------------------------
# decl_spec: the single source of truth for the DecL->kwargs mapping
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind, numbers, expected",
    [
        ("ph", [0.5], {"name": "ph", "a": 0.5}),
        ("wang", [0.3], {"name": "wang", "lam": 0.3}),
        ("dual", [2.5], {"name": "dual", "b": 2.5}),
        ("tvar", [0.99], {"name": "tvar", "p": 0.99}),
        ("ccoc", [0.25], {"name": "ccoc", "r": 0.25}),   # r, not the discount d
        ("roe", [0.1], {"name": "ccoc", "r": 0.1}),       # legacy alias
        ("bitvar", [0.9, 0.99, 0.5], {"name": "bitvar", "p0": 0.9, "p1": 0.99, "w1": 0.5}),
        ("power", [0.01, 1.0, 2], {"name": "power", "x0": 0.01, "x1": 1.0, "alpha": 2}),
        ("beta", [0.5, 2], {"name": "beta", "a": 0.5, "b": 2}),
        ("cll", [1.5], {"name": "cll", "b": 1.5}),
        ("clin", [0.3], {"name": "clin", "slope": 0.3}),
        ("lep", [0.1], {"name": "lep", "r": 0.1}),
        ("ly", [0.1], {"name": "ly", "r": 0.1}),
    ],
)
def test_decl_spec_mapping(kind, numbers, expected):
    assert Distortion.decl_spec(kind, numbers) == expected


def test_decl_spec_rejects_variadic_and_combinators():
    # wtdtvar takes parameter vectors; minimum/mixture take distortion refs.
    for kind in ("wtdtvar", "minimum", "mixture"):
        with pytest.raises(ValueError, match="no DecL number-list form"):
            Distortion.decl_spec(kind, [0.5])


def test_decl_spec_rejects_arity_mismatch():
    with pytest.raises(ValueError, match="expects 1 parameter"):
        Distortion.decl_spec("ph", [0.5, 0.6])
    with pytest.raises(ValueError, match="expects 3 parameter"):
        Distortion.decl_spec("bitvar", [0.9, 0.99])


def test_decl_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown distortion kind"):
        Distortion.decl_spec("frobnicate", [1])


# ----------------------------------------------------------------------
# End-to-end build() through the new grammar
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "program, subclass",
    [
        ("distortion D ph 0.5", "PHDistortion"),
        ("distortion D ccoc .25", "CCoCDistortion"),
        ("distortion D dual 2.5", "DualDistortion"),
        ("distortion D tvar 0.99", "TVaRDistortion"),
        ("distortion D bitvar 0.9 0.99 0.5", "BiTVaRDistortion"),
        ("distortion D power 0.01 1.0 2", "PowerDistortion"),
        ("distortion D beta 0.5 2", "BetaDistortion"),
    ],
)
def test_build_distortion(program, subclass):
    d = build(program)
    assert type(d).__name__ == subclass
    # endpoint contract: g(0)=0, g(1)=1
    assert d.g(0.0) == pytest.approx(0.0, abs=1e-12)
    assert d.g(1.0) == pytest.approx(1.0, abs=1e-12)


def test_build_ccoc_uses_r_not_discount():
    d = build("distortion D ccoc 0.25")
    assert d.r == pytest.approx(0.25)


def test_build_wtdtvar_rejected_in_decl():
    with pytest.raises(Exception, match="no DecL number-list form"):
        build("distortion D wtdtvar 0.9 0.5")
