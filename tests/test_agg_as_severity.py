"""[Agg-As-Severity]: an agg or port reference as a severity in DecL (1.0.0a291).

``sev agg.NAME`` / ``sev port.NAME`` (and the ``ssev`` and ``!`` forms) let an
aggregate already in the recipe base serve as the severity of another
aggregate. The motivating shape is a US personal auto split limit: 100 per
claimant on the inner, 300 per policy on the outer.

The reference is **resolved at build time**, which is what makes it DecL's
first deferred reference: the severity is the inner's *computed output*, which
exists only after an update. Everything else about it follows the fully formed
``dsev`` model of ``dev/done/plan-agg-port-as-sev.md`` section 2.

The programmatic half (``Severity(agg)`` / ``as_severity``) is tested in
``tests/test_reference_severity.py``; both share
``aggregate._severity._dhistogram_from_object``.

Programs are mirrored in ``src/aggregate/agg/decl-testers.agg`` section ASV, and
the shipped-corpus form is ``_test_suite.agg`` section Z.
"""

import logging
import warnings

import numpy as np
import pytest

from aggregate import build
from aggregate.underwriter import Underwriter

logging.disable(logging.CRITICAL)

HINTS = 'hints{log2=16; bs=1/32}'

#: The three inners of the plan's section 5 discussion, identical except for
#: the frequency clause, plus the aggregate-reinsurance benchmark.
INNERS = f"""
agg AAS.SL    1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt {HINTS}

agg AAS.SLzt  1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt ! {HINTS}

agg AAS.SL2   1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson {HINTS}

agg AAS.SL2re 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson aggregate ceded to 300 xs 0 {HINTS}
"""


@pytest.fixture(scope='module', autouse=True)
def _inners():
    """Register the shared inners once for the module."""
    build.build_many(INNERS)


def _law(a):
    """The (xs, p) law of an aggregate, trimmed to its common grid."""
    return a.xs, a.agg_density


def _same_law(a, b, atol=1e-9):
    """Two aggregates on the same grid carry the same density."""
    assert a.bs == b.bs and a.log2 == b.log2
    return np.allclose(a.agg_density, b.agg_density, atol=atol)


# ----------------------------------------------------------------------
# Exactness: the reference really is the inner's law
# ----------------------------------------------------------------------
def test_two_claims_of_a_reference_is_the_self_convolution():
    build('agg AAS.Dice dfreq [1] dsev [1 2 3] hints{log2=8; bs=1}')
    outer = build('agg AAS.TwoDice dfreq [2] sev agg.AAS.Dice')
    p = np.array([0, 1 / 3, 1 / 3, 1 / 3])
    want = np.convolve(p, p)
    got = outer.agg_density[:len(want)]
    assert np.allclose(got, want, atol=1e-12)
    assert np.allclose(outer.agg_density[len(want):], 0, atol=1e-12)


def test_reference_severity_moments_are_the_inners_exact_atom_sums():
    src = build(f'agg AAS.MRef 1.5 claims 100 xs 0 sev gamma 50 cv 2 '
                f'poisson zt {HINTS}')
    outer = build('agg AAS.OneOf dfreq [1] sev agg.AAS.MRef')
    sev = outer.sevs[0]
    xs, ps = src.xs, src.agg_density / src.agg_density.sum()
    assert sev.sev1 == pytest.approx(float(np.sum(xs * ps)), rel=1e-10)
    assert sev.sev2 == pytest.approx(float(np.sum(xs ** 2 * ps)), rel=1e-10)


def test_port_reference_matches_the_portfolio_total():
    # a portfolio's own trailer sits before its units (``port_out: PORT name
    # as_label trailer agg_list``), which is where the hygiene rule reads it
    p = build(f'port AAS.PBook {HINTS} agg AAS.U1 1.5 claims 100 xs 0 '
              f'sev gamma 50 cv 2 poisson zt')
    ref = build('agg AAS.FromPort dfreq [1] sev port.AAS.PBook')
    xs = p.density_df.loss.values
    ps = p.density_df.p_total.values
    ps = ps / ps.sum()
    sev = ref.sevs[0]
    assert sev.sev1 == pytest.approx(float(np.sum(xs * ps)), rel=1e-9)
    assert sev.sev2 == pytest.approx(float(np.sum(xs ** 2 * ps)), rel=1e-9)


# ----------------------------------------------------------------------
# The split limit: three representations, and which pairs coincide
# ----------------------------------------------------------------------
def test_severity_bang_equals_the_aggregate_reinsurance_benchmark():
    # 5,000 policies, ~22% of which pay nothing: the zero atom is kept by the
    # severity-side ``!``, and the 300 cap is an ordinary occurrence limit.
    a = build('agg AAS.SplitA 5000 claims 300 xs 0 sev agg.AAS.SL2 ! poisson')
    # the same law written with aggregate reinsurance on the inner and no
    # outer layers clause
    b = build('agg AAS.SplitB 5000 claims sev agg.AAS.SL2re poisson')
    # The severity IS the same law: all three theoretical moments agree.
    for ma, mb in zip(a.sevs[0].moms(), b.sevs[0].moms()):
        assert ma == pytest.approx(mb, rel=1e-11)
    # The realized aggregates differ only by discretization, and by a knowable
    # amount: an unlayered reference takes the mean-preserving linear scatter
    # (b reproduces the severity mean exactly), while the outer layers clause
    # on a puts it on the sf-difference path, which carries a small positional
    # bias that shrinks with the grid. This is the ordinary ``dsev`` behavior,
    # and it is the numeric reason the docs prefer the occurrence-limit form.
    assert b.est_sev_m == pytest.approx(b.sevs[0].moms()[0], rel=1e-12)
    assert a.est_m == pytest.approx(b.est_m, rel=1e-3)


def test_the_zt_identity():
    # conditioning the per-policy aggregate on being positive is the same as
    # zero-truncating the frequency, for an almost surely positive severity
    a = build('agg AAS.ZtA 5000 claims 300 xs 0 sev agg.AAS.SL2 poisson')
    b = build('agg AAS.ZtB 5000 claims 300 xs 0 sev agg.AAS.SLzt poisson')
    assert _same_law(a, b, atol=1e-9)


def test_the_freq_bang_variant_is_a_different_model():
    # ``poisson zt`` holds the truncated mean at 1.5; ``poisson zt !`` makes
    # 1.5 the underlying parameter and the truncated mean about 1.93
    a = build('agg AAS.ZtA2 5000 claims 300 xs 0 sev agg.AAS.SL2 poisson')
    c = build('agg AAS.ZtC 5000 claims 300 xs 0 sev agg.AAS.SL poisson')
    assert not np.allclose(a.agg_density, c.agg_density, atol=1e-6)
    assert a.est_m != pytest.approx(c.est_m, rel=1e-3)


def test_the_headline_program_builds():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg AAS.Auto 5000 claims 300 xs 0 sev agg.AAS.SL ! '
                  'mixed gamma .2')
    assert a.agg_density is not None
    # 5,000 policies of a per-policy law with mean ~44.3, capped at 300 (which
    # bites a little, so the outer mean sits just under 5000 x 44.3)
    assert 0.95 * 5000 * 44.3 < a.est_m <= 5000 * 44.4


def test_only_the_plain_inner_carries_a_genuine_zero_atom():
    # ``P(S = 0)`` is exactly ``e**-1.5`` for the plain Poisson inner and
    # exactly 0 for the zero-truncated one. What the zt inner does still carry
    # in its FIRST BUCKET is discretization, not a zero atom: a gamma with
    # cv 2 has shape 0.25 and puts ~10% of one claim below bs/2.
    zt = build('agg AAS.ZtInner dfreq [1] sev agg.AAS.SL')
    plain = build('agg AAS.PlainInner dfreq [1] sev agg.AAS.SL2 !')
    assert float(plain.sevs[0].sev_ps[0]) > np.exp(-1.5)
    assert float(zt.sevs[0].sev_ps[0]) < np.exp(-1.5)


def test_conditioning_a_zero_truncated_reference_warns():
    # The trap the plan did not anticipate: a zt inner has no zero atom in
    # theory but does in the materialized dsev, so a default-conditional layers
    # clause quietly rescales it away and lifts the severity mean.
    with pytest.warns(UserWarning, match='conditions on exceeding'):
        cond = build('agg AAS.Cond 5000 claims 300 xs 0 sev agg.AAS.SL poisson')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        uncond = build('agg AAS.Uncond 5000 claims 300 xs 0 sev agg.AAS.SL ! '
                       'poisson')
    assert cond.sevs[0].moms()[0] > uncond.sevs[0].moms()[0]


def test_conditioning_a_genuine_zero_atom_is_silent():
    # a plain Poisson inner really does have P(S = 0) = e**-1.5; conditioning
    # it away is the documented "5,000 loss-bearing policies" model
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        build('agg AAS.CondOk 5000 claims 300 xs 0 sev agg.AAS.SL2 poisson')


def test_no_layers_clause_keeps_the_zero_atom_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        a = build('agg AAS.NoLayer 5000 claims sev agg.AAS.SL2 poisson')
    assert float(a.sevs[0].sev_ps[0]) > np.exp(-1.5)


# ----------------------------------------------------------------------
# Hygiene: a referenced inner must pin its own grid
# ----------------------------------------------------------------------
def test_reference_to_an_unhinted_inner_raises_and_names_with_hints():
    build('agg AAS.Loose 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson')
    with pytest.raises(ValueError) as e:
        build('agg AAS.UsesLoose 10 claims sev agg.AAS.Loose poisson')
    msg = str(e.value)
    assert 'with_hints' in msg
    assert 'hints{log2=' in msg and 'bs=' in msg


def test_the_pasted_hints_from_the_error_actually_work():
    build('agg AAS.Loose2 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson')
    with pytest.raises(ValueError) as e:
        build('agg AAS.UsesLoose2 10 claims sev agg.AAS.Loose2 poisson')
    hint = str(e.value)[str(e.value).index('hints{'):]
    hint = hint[:hint.index('}') + 1]
    build(f'agg AAS.Loose2 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson {hint}')
    a = build('agg AAS.UsesLoose2 10 claims sev agg.AAS.Loose2 poisson')
    assert a.agg_density is not None


def test_partial_hints_still_raise():
    build('agg AAS.Half 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson '
          'hints{log2=16}')
    with pytest.raises(ValueError, match='with_hints'):
        build('agg AAS.UsesHalf 10 claims sev agg.AAS.Half poisson')


# ----------------------------------------------------------------------
# with_hints: the certification helper
# ----------------------------------------------------------------------
def test_with_hints_round_trips_and_certifies():
    a = build('agg AAS.Cert 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson')
    prog = a.with_hints()
    assert 'hints{' in prog and 'log2=' in prog and 'bs=' in prog
    build(prog)
    outer = build('agg AAS.UsesCert 10 claims sev agg.AAS.Cert poisson')
    assert outer.sevs[0].sev1 == pytest.approx(a.est_m, rel=1e-9)


def test_with_hints_takes_extra_keywords():
    a = build('agg AAS.Cert2 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson')
    prog = a.with_hints(normalize=False)
    assert 'normalize=False' in prog


def test_with_hints_replaces_rather_than_duplicates():
    a = build('agg AAS.Cert3 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson '
              'hints{log2=14; padding=2}')
    prog = a.with_hints()
    assert prog.count('hints{') == 1               # one clause, always
    assert 'padding=2' in prog                     # the author's other hints survive
    assert f'log2={a.log2}' in prog                # the realized grid, wherever it landed
    assert 'bs=' in prog                           # and the bs the clause never named


def test_with_hints_on_a_portfolio():
    p = build('port AAS.CertP agg AAS.CU 1.5 claims 100 xs 0 '
              'sev gamma 50 cv 2 poisson')
    prog = p.with_hints()
    assert 'hints{' in prog and 'log2=' in prog and 'bs=' in prog


def test_with_hints_needs_an_updated_object():
    a = build('agg AAS.CertNo 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson',
              update=False)
    with pytest.raises(ValueError):
        a.with_hints()


# ----------------------------------------------------------------------
# Resolution mechanics: depth, cycles, iteration, algebra
# ----------------------------------------------------------------------
def test_depth_two_chain():
    build(f'agg AAS.D0 dfreq [1] dsev [1 2 3] {HINTS}')
    build(f'agg AAS.D1 dfreq [2] sev agg.AAS.D0 {HINTS}')
    a = build('agg AAS.D2 dfreq [2] sev agg.AAS.D1')
    assert a.est_m == pytest.approx(8.0, rel=1e-9)


def test_reference_cycle_raises_with_the_chain_named():
    uw = Underwriter()
    uw.build(f'agg CY.A dfreq [1] dsev [1] {HINTS}')
    uw.build(f'agg CY.B dfreq [1] sev agg.CY.A {HINTS}')
    # redefine A to point back at B: A -> B -> A. The cycle is caught on the
    # redefining build itself, since resolution runs at construction.
    with pytest.raises(ValueError, match='severity reference cycle') as e:
        uw.build(f'agg CY.A dfreq [1] sev agg.CY.B {HINTS}')
    assert 'agg.CY.B -> agg.CY.A -> agg.CY.B' in str(e.value)
    # and again from a third party, because the cyclic recipe is registered
    with pytest.raises(ValueError, match='severity reference cycle'):
        uw.build('agg CY.C dfreq [1] sev agg.CY.A')


def test_self_reference_raises():
    uw = Underwriter()
    uw.build(f'agg SR.A dfreq [1] dsev [1] {HINTS}')
    with pytest.raises(ValueError, match='severity reference cycle') as e:
        uw.build(f'agg SR.A dfreq [1] sev agg.SR.A {HINTS}')
    assert 'agg.SR.A -> agg.SR.A' in str(e.value)


def test_unknown_reference_raises_at_parse_time():
    with pytest.raises((LookupError, ValueError)):
        build('agg AAS.Missing 10 claims sev agg.AAS.NoSuchThing poisson')


def test_algebra_on_a_reference_carrier_raises():
    build(f'agg AAS.Carrier dfreq [1] sev agg.AAS.SL {HINTS}')
    with pytest.raises(ValueError, match='cannot be scaled'):
        build('agg AAS.Scaled 2 * agg.AAS.Carrier')
    with pytest.raises(ValueError, match='cannot be shifted'):
        build('agg AAS.Shifted agg.AAS.Carrier + 10')
    with pytest.raises(ValueError, match='cannot be shifted'):
        build('agg AAS.Shifted2 agg.AAS.Carrier - 10')


def test_inhomogeneous_scaling_of_a_reference_carrier_is_legal():
    build(f'agg AAS.Carrier2 5 claims sev agg.AAS.SL poisson {HINTS}')
    a = build('agg AAS.IScaled 2 @ agg.AAS.Carrier2')
    assert a.n == pytest.approx(10.0)


def test_iterated_builds_re_resolve_the_inner():
    uw = Underwriter()
    uw.build(f'agg IT.In dfreq [1] dsev [1] {HINTS}')
    a1 = uw.build('agg IT.Out dfreq [1] sev agg.IT.In')
    assert a1.est_m == pytest.approx(1.0, rel=1e-9)
    # redefine the inner; the outer re-resolves on the next build
    uw.build(f'agg IT.In dfreq [1] dsev [5] {HINTS}')
    a2 = uw.build('agg IT.Out dfreq [1] sev agg.IT.In')
    assert a2.est_m == pytest.approx(5.0, rel=1e-9)


def test_one_program_can_define_and_use_an_inner():
    uw = Underwriter()
    rv = uw.build_many(f"""
agg SP.SL 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt {HINTS}

agg SP.Auto 5000 claims 300 xs 0 sev agg.SP.SL mixed gamma .2
""")
    assert len(rv) == 2
    inner, outer = rv[0].object, rv[1].object
    assert inner.agg_density is not None and outer.agg_density is not None
    # the inner's hints must NOT leak onto the outer's grid
    assert outer.bs != inner.bs


def test_hints_do_not_leak_between_statements():
    uw = Underwriter()
    rv = uw.build_many("""
agg HL.Hinted 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson hints{log2=16; bs=1/32}

agg HL.Plain 5000 claims 300 xs 0 sev gamma 50 cv 2 poisson
""")
    hinted, plain = rv[0].object, rv[1].object
    assert (hinted.log2, hinted.bs) == (16, 1 / 32)
    assert plain.bs != 1 / 32


# ----------------------------------------------------------------------
# Signed matrix
# ----------------------------------------------------------------------
def test_ssev_over_a_signed_inner_keeps_the_negative_atoms():
    build(f'agg AAS.Signed dfreq [1] ssev 100 - lognorm 50 cv .5 {HINTS}')
    a = build('agg AAS.UsesSigned dfreq [1] ssev agg.AAS.Signed')
    assert a.sevs[0].support_atoms.min() < 0
    assert a.sevs[0].signed is True


def test_sev_over_a_signed_inner_warns_and_clamps():
    build(f'agg AAS.Signed2 dfreq [1] ssev 100 - lognorm 50 cv .5 {HINTS}')
    with pytest.warns(UserWarning, match='clamped onto the zero atom'):
        a = build('agg AAS.UsesSigned2 dfreq [1] sev agg.AAS.Signed2')
    assert a.sevs[0].support_atoms.min() >= 0


def test_ssev_over_a_nonnegative_inner_is_silent():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        build('agg AAS.UsesPlainSsev dfreq [1] ssev agg.AAS.SL')
    assert not [w for w in rec if 'clamped' in str(w.message)]


# ----------------------------------------------------------------------
# Tail reporting: the inner's THEORETICAL tail, numerics untouched
# ----------------------------------------------------------------------
def _right_tail(a):
    return a.tail_behavior_df.loc['aggregate', 'right_tail']


def _agg_max(a):
    return a.tail_behavior_df.loc['aggregate', 'max']


def test_unbounded_inner_reports_an_unbounded_outer_tail():
    build(f'agg AAS.UB dfreq [1] sev gamma 100 cv 1 {HINTS}')
    a = build('agg AAS.UsesUB dfreq [1] sev agg.AAS.UB')
    assert not np.isfinite(_agg_max(a))
    assert _right_tail(a) != 'bounded'


def test_the_descriptor_is_transitive():
    build(f'agg AAS.UB2 dfreq [1] sev gamma 100 cv 1 {HINTS}')
    build(f'agg AAS.UBOuter dfreq [1] sev agg.AAS.UB2 {HINTS}')
    a = build('agg AAS.UBOuter2 dfreq [1] sev agg.AAS.UBOuter')
    assert not np.isfinite(_agg_max(a))


def test_a_finite_outer_limit_overrides_the_descriptor():
    build(f'agg AAS.UB3 dfreq [1] sev gamma 100 cv 1 {HINTS}')
    a = build('agg AAS.UsesUB3 dfreq [1] 500 xs 0 sev agg.AAS.UB3')
    assert np.isfinite(_agg_max(a))
    assert _right_tail(a) == 'bounded'


def test_a_ceded_to_aggregate_cover_inner_reports_bounded():
    a = build('agg AAS.UsesCeded dfreq [1] sev agg.AAS.SL2re')
    assert np.isfinite(_agg_max(a))


def test_a_net_of_inner_reports_unbounded():
    build('agg AAS.NetInner 1.5 claims sev gamma 50 cv 2 poisson '
          f'aggregate net of 300 xs 0 {HINTS}')
    a = build('agg AAS.UsesNet dfreq [1] sev agg.AAS.NetInner')
    assert not np.isfinite(_agg_max(a))


def test_a_bounded_severity_under_a_poisson_frequency_is_unbounded():
    build(f'agg AAS.BsPo 5 claims dsev [1 2 3] poisson {HINTS}')
    a = build('agg AAS.UsesBsPo dfreq [1] sev agg.AAS.BsPo')
    assert not np.isfinite(_agg_max(a))


def test_a_genuinely_bounded_inner_reports_bounded():
    build(f'agg AAS.Bnd dfreq [2] dsev [1 2 3] {HINTS}')
    a = build('agg AAS.UsesBnd dfreq [1] sev agg.AAS.Bnd')
    assert np.isfinite(_agg_max(a))
    assert _right_tail(a) == 'bounded'


def test_the_descriptor_does_not_reach_the_bucket_sizer():
    # numerics ride the materialized dsev: the grid an unbounded-descriptor
    # reference picks is the grid its atoms pick
    build(f'agg AAS.UB4 dfreq [1] sev gamma 100 cv 1 {HINTS}')
    a = build('agg AAS.UsesUB4 dfreq [1] sev agg.AAS.UB4')
    left, right = a._loss_tail_classes()
    from aggregate import tail as _tail
    assert not _tail.is_thick(right)
    # the bounded-severity window still applies, i.e. the sizer still sees the
    # finite atoms it is going to convolve
    assert bool(a.bs_window_df.loc['bounded_small', 'applies'])


def test_bounded_follows_the_referenced_object_not_its_atoms():
    # author ruling: a reference to an unbounded aggregate reports unbounded,
    # on every surface, however finite the atoms it materialized to
    build(f'agg AAS.UB5 dfreq [1] sev gamma 100 cv 1 {HINTS}')
    a = build('agg AAS.UsesUB5 dfreq [1] sev agg.AAS.UB5')
    assert a.bounded is False
    assert a.sevs[0].bounded is False
    assert not np.isfinite(_agg_max(a))          # and the frame agrees


def test_a_bounded_reference_still_reports_bounded():
    build(f'agg AAS.Bnd2 dfreq [2] dsev [1 2 3] {HINTS}')
    a = build('agg AAS.UsesBnd2 dfreq [1] sev agg.AAS.Bnd2')
    assert a.bounded is True
    assert a.sevs[0].bounded is True
    assert np.isfinite(_agg_max(a))


def test_both_routes_into_a_reference_report_the_same_boundedness():
    from aggregate.distributions import Severity
    src = build('agg AAS.UB6 dfreq [1] sev gamma 100 cv 1')
    programmatic = Severity(src)
    build(f'agg AAS.UB7 dfreq [1] sev gamma 100 cv 1 {HINTS}')
    declarative = build('agg AAS.UsesUB7 dfreq [1] sev agg.AAS.UB7').sevs[0]
    assert programmatic.bounded is declarative.bounded is False


# ----------------------------------------------------------------------
# Round trip and interaction with the rest of the surface
# ----------------------------------------------------------------------
def test_the_unparser_renders_the_reference_back():
    a = build('agg AAS.Rt 10 claims sev agg.AAS.SL poisson')
    assert 'sev agg.AAS.SL' in a.format_program(layout='terse')


def test_the_unparser_renders_ssev_and_the_bang():
    build(f'agg AAS.Signed3 dfreq [1] ssev 100 - lognorm 50 cv .5 {HINTS}')
    a = build('agg AAS.RtS dfreq [1] ssev agg.AAS.Signed3')
    assert 'ssev agg.AAS.Signed3' in a.format_program(layout='terse')
    b = build('agg AAS.RtB 5000 claims 300 xs 0 sev agg.AAS.SL2 ! poisson')
    assert 'sev agg.AAS.SL2 !' in b.format_program(layout='terse')


def test_normalize_false_on_the_inner_keeps_the_deficit():
    build('agg AAS.Def 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson '
          'hints{log2=8; bs=1/32; normalize=False}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = build('agg AAS.UsesDef dfreq [1] sev agg.AAS.Def')
    assert float(np.sum(a.sevs[0].sev_ps)) < 1 - 1e-6


def test_approximate_over_a_reference_builds():
    a = build('agg AAS.Approx 5000 claims sev agg.AAS.SL poisson '
              'approximate slognorm')
    assert a.approximation == 'slognorm'
    assert a.agg_density is not None


def test_create_frequency_on_a_reference_carrier():
    a = build('agg AAS.CF 100 claims sev agg.AAS.SL poisson')
    f = a.create_frequency()
    assert f.est_m == pytest.approx(100.0, rel=1e-3)


def test_pnl_engine_over_a_reference_severity():
    p = build('pnl AAS.PnL 300000 premium less '
              'agg AAS.PnLEngine 5000 claims sev agg.AAS.SL poisson')
    assert p is not None


# ----------------------------------------------------------------------
# [Agg-As-Severity-Commensurable-Grids] (a292): the outer grid and the
# reference's own lattice must never be incommensurable
# ----------------------------------------------------------------------
#: A reference whose lattice is NOT a power of two, which is what it takes to
#: reach the snap at all: every bucket the estimator can pick under 1 is a power
#: of two and every one at or above 1 is an integer, so a dyadic ``d`` is
#: commensurable with all of them already.
ODD = 'agg AAS.Odd 2 claims sev gamma 3 cv .5 poisson hints{log2=12; bs=1/3}'


def _commensurable(b0, d):
    r = max(b0, d) / min(b0, d)
    return np.isclose(r, round(r), rtol=1e-9)


def test_a_dyadic_reference_lattice_needs_no_snap():
    # bs=1/32 divides every bucket the estimator can choose, so the property
    # holds without the snap firing at all. Assert the property, not the path.
    a = build('agg AAS.Comm 500 claims sev agg.AAS.SL poisson')
    assert _commensurable(a.bs, 1 / 32)
    assert a._bs_snap is None


def test_an_estimate_below_the_lattice_snaps_up_to_it():
    build(ODD)
    a = build('agg AAS.OddOut dfreq [2] sev agg.AAS.Odd')
    assert a.bs == pytest.approx(1 / 3, rel=1e-12)
    assert a._bs_snap is not None
    assert a._bs_snap['bs_before'] < 1 / 3
    assert a._bs_snap['m'] == 0                 # never below d
    assert a._bs_snap['ref'] == 'agg.AAS.Odd'


def test_the_snap_is_named_in_the_bs_explanation():
    build(ODD)
    a = build('agg AAS.OddOut2 dfreq [2] sev agg.AAS.Odd')
    text = a.bs_explanation
    assert 'agg.AAS.Odd' in text
    assert '1/3' in text                        # the lattice and the new bs
    assert 'commensurable' in text


def test_the_snapped_grid_still_covers_its_window():
    # the snap re-derives origin and log2 through the one sizing kernel, so the
    # realized grid must still cover the winning method's window
    build(ODD)
    a = build('agg AAS.OddOut3 dfreq [2] sev agg.AAS.Odd')
    df = a.bs_window_df
    won = df[df['selected'].astype(bool)].iloc[0]
    top = float(won['x_min']) + (1 << int(won['log2'])) * float(won['bs'])
    assert top >= float(won['x_max'])
    assert float(df.loc['used', 'bs']) == pytest.approx(a.bs, rel=1e-12)


def test_an_integer_lattice_reference_keeps_the_exact_discrete_window():
    build('agg AAS.D1 dfreq [1] dsev [1 2 3] hints{log2=8; bs=1}')
    a = build('agg AAS.D2 dfreq [2] sev agg.AAS.D1')
    assert a.bs == 1                            # b0 / d == 1, already commensurable
    assert a._bs_snap is None
    assert bool(a.bs_window_df.loc['exact_discrete', 'selected'])


def test_an_exact_integer_multiple_is_left_alone():
    # no forced power of two when the estimator already landed on a multiple
    build('agg AAS.Ten 5 claims dsev [10 20 30] poisson hints{log2=10; bs=10}')
    a = build('agg AAS.TenOut dfreq [2] sev agg.AAS.Ten')
    assert _commensurable(a.bs, 10)
    assert a._bs_snap is None


def test_a_pinned_incommensurable_bs_warns_and_is_honored():
    build(ODD)
    with pytest.warns(UserWarning, match='incommensurable') as rec:
        a = build('agg AAS.OddPin dfreq [2] sev agg.AAS.Odd', bs=0.5)
    assert a.bs == 0.5                          # the pin is honored, no back door
    assert a._bs_snap is None
    msg = ' '.join(str(w.message) for w in rec)
    assert '1/3' in msg and '0.666667' in msg   # the adjacent multiples of d


def test_a_pinned_commensurable_bs_is_silent():
    build(ODD)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        a = build('agg AAS.OddPinOk dfreq [2] sev agg.AAS.Odd', bs=2 / 3)
    assert a.bs == pytest.approx(2 / 3, rel=1e-12)
    assert not [w for w in rec if 'incommensurable' in str(w.message)]


# ----------------------------------------------------------------------
# [Agg-As-Severity-Port-Units] (a293): a portfolio unit may take its
# severity from a reference
# ----------------------------------------------------------------------
def test_a_portfolio_unit_resolves_a_reference_severity():
    p = build("""port AAS.UnitBook
    agg AAS.UnitA 500 claims sev agg.AAS.SL poisson
    agg AAS.UnitB 10 claims sev lognorm 50 cv 1 poisson""")
    unit = p.agg_list[0]
    assert unit.sevs[0].sev_kind == 'dhistogram'
    assert unit.sevs[0].sev1 == pytest.approx(
        build('agg AAS.SL 1.5 claims 100 xs 0 sev gamma 50 cv 2 '
              f'poisson zt {HINTS}').est_m, rel=1e-9)
    assert p.est_m > 0


def test_a_portfolio_unit_carries_the_reference_provenance():
    p = build("""port AAS.UnitBook2
    agg AAS.UnitC 500 claims sev agg.AAS.SL poisson
    agg AAS.UnitD 10 claims sev lognorm 50 cv 1 poisson""")
    sev = p.agg_list[0].sevs[0]
    assert sev.reference_id == 'agg.AAS.SL'
    assert sev.reference_bs == pytest.approx(1 / 32)
    # the unit's own tail report reads the source's theoretical tail
    assert not np.isfinite(p.agg_list[0].tail_behavior_df.loc['aggregate', 'max'])
    # and the unit that carries no reference is untouched
    assert p.agg_list[1].sevs[0].reference_id == ''


def test_a_portfolio_unit_reference_round_trips():
    p = build("""port AAS.UnitBook3
    agg AAS.UnitE 500 claims sev agg.AAS.SL poisson
    agg AAS.UnitF 10 claims sev lognorm 50 cv 1 poisson""")
    assert 'sev agg.AAS.SL' in p.format_program(layout='terse')


def test_an_unhinted_reference_in_a_portfolio_unit_raises():
    build('agg AAS.LooseP 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson')
    with pytest.raises(ValueError, match='with_hints'):
        build("""port AAS.UnitBad
    agg AAS.UnitG 500 claims sev agg.AAS.LooseP poisson
    agg AAS.UnitH 10 claims sev lognorm 50 cv 1 poisson""")
