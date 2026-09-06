"""Every entry in the shipped library builds, and the invariants it claims hold.

``tests/test_agg_libraries.py`` checks that each entry **parses** and that the
file's filing conventions hold. This module checks that each entry **builds**,
which is what ``[Agg-Library-Build-Check]`` in ``dev/TODO.md`` asked for: an
entry could parse cleanly and still produce garbage moments with no test signal
at all.

It also carries the seven invariants that used to live in ``doc{{{...}}}``
bodies inside ``library.agg`` and ran through ``Recipe.run``. Executing prose
was the wrong mechanism (``dev/plan-decommission-docs.md``): the code was
invisible to ruff, to editors and to tracebacks, which reported
``<recipe LayerPicks:check>`` and no line in any real file. The invariants
themselves were never the problem, so they are transcribed here unchanged in
substance, one test function each, with the entry they check named in the
function name. The long-form write-ups they came from are now notes in
``aggregate-presentations``.

Three baselines make a **new** failure a finding rather than noise:
``CANNOT_BUILD`` for entries that deliberately refuse to build, and
``VALIDATION_BASELINE`` for entries that deliberately do not clear validation.
Both are asserted in each direction, so an entry that starts passing is as much
a finding as one that starts failing.
"""
import numpy as np
import pandas as pd
import pytest

from aggregate import Underwriter, build
from aggregate.constants import Validation

# ----------------------------------------------------------------------
# The library, loaded once
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def library():
    uw = Underwriter(databases='library')
    uw.load()
    return uw


def _entries():
    """(kind, name) for every entry in the shipped library."""
    uw = Underwriter(databases='library')
    uw.load()
    return sorted(uw._recipes)


ENTRIES = _entries()

#: Entries whose build is expensive enough to quarantine behind ``slow``.
#: Measured at 1.0.0a299 (the first six) and re-measured at 1.0.0a327 when
#: the new-examples merge landed: everything over a second sits here.
#: ``BivariateNormal`` alone is twelve seconds; the capstone chain re-derives
#: its five-band picked book at every link, so the whole chain is quarantined.
SLOW_ENTRIES = {
    'BivariateNormal', 'BivariateCatPair', 'BivariateClaytonMixed',
    'BivariateGumbel', 'BivariateIndependent', 'LayerPicks.Uniform',
    # the capstone chain, 1.2s to 4.4s each
    'Capstone.Gross', 'Capstone.ExposureRating', 'Capstone.SelectedLosses',
    'Capstone.LossPicksTest', 'Capstone.XOL', 'Capstone.XOL.PnL',
    'Capstone.FullProgram', 'Capstone.PnL', 'Capstone.PC',
    'Capstone.GrossNet',
    # renewal counts convolve one wait law per arrival
    'RenewalExponentialWait', 'RenewalLayeredWait',
    # the clash solver builds both marginals plus the joint
    'BivariateClash',
    # heavy lomax books under a 20000 limit, and a cv 3 tower
    'ReinstatementTreaty', 'NoReinstatementTreaty', 'ReinstatementNumberWords',
    'CedingCommission', 'TowerLimitProfile', 'PnLDerive',
}

#: Entries that deliberately **refuse** to build, and the exception each
#: raises. Every one is a demonstration of a refusal rather than a defect, so
#: the refusal is the thing worth pinning: a mixture severity has no standalone
#: meaning, a P&L unit inside a portfolio is deferred work, and an
#: infinite-variance aggregate cannot have a bucket size estimated for it.
CANNOT_BUILD = {
    # No finite second moment, so the bucket sizer has nothing to work with.
    # The entry exists to show the refusal and the message that explains it.
    'DefectivePareto': 'InfiniteVarianceError',
    # A mixture severity is not a distribution on its own; it has to be folded
    # into an aggregate. Both entries are severity *components*, used by name.
    # ``ISOMixedExponential`` carried the first of these until the library
    # reorganization renamed it ``CommAutoMixedExponentialSev``.
    'CommAutoMixedExponentialSev': 'CannotBuild',
    'MixedExponentialSev': 'CannotBuild',
    # Deferred by design: book-level P&L raises with the reason rather than
    # producing something wrong. ``BivariatePnLAxis`` and ``NumericsPayPair``
    # were the other two demonstrations and left the library in the same
    # reorganization, with no successor.
    'PnLBook': 'NotImplementedError',
}

#: Entries that build but do not clear :attr:`Validation.passes`, with the
#: exact flags each reports. Validation compares the FFT estimate against the
#: analytic moments of the declared severity, so an entry fails it whenever the
#: declaration deliberately leaves those moments behind or cannot represent
#: them on any practical grid. Grouped by cause.
VALIDATION_BASELINE = {
    # --- Heavy-tail reference curves -----------------------------------
    # The severity curve reference exhibits each family at parameters that
    # show its shape, which for the power-law families means an infinite or
    # near-infinite higher moment the discretized grid cannot reproduce.
    # These are reference entries, not models to copy.
    'CurveBetaPrime': 'SEV_SKEW|AGG_SKEW',
    'CurveBurr12': 'SEV_SKEW|AGG_SKEW',
    'CurveBurrIII': 'SEV_CV|AGG_CV',
    'CurveFrechet': 'SEV_CV|AGG_CV',
    'CurveGPD': 'SEV_MEAN|SEV_SKEW|AGG_MEAN|AGG_SKEW',
    'CurveInverseGamma': 'SEV_SKEW|AGG_SKEW',
    'CurveLogLogistic': 'SEV_CV|AGG_CV',
    'CurveLomax': 'SEV_MEAN|SEV_CV|AGG_MEAN|AGG_CV',
    'CurvePareto': 'SEV_MEAN|SEV_CV|AGG_MEAN|AGG_CV|INFEASIBLE',
    'SevOneParameterScaled': 'SEV_MEAN|AGG_MEAN',
    # --- Failure demonstrations ----------------------------------------
    # These exist *to* fail. HeavyTailValidation is the worked example of what
    # a failed validation looks like and what the explanation says.
    # The reorganization respelled this one from ``100 * pareto 1.3 + -100``
    # to the equivalent ``100 * lomax 1.3``. Same law, but the sizer reads the
    # shifted Pareto's body differently, so INFEASIBLE no longer fires and the
    # entry now reports three flags rather than four.
    'HeavyTailValidation': 'SEV_MEAN|AGG_MEAN|DEFECTIVE',
    'InverseGaussianMixed': 'AGG_MEAN|AGG_CV|AGG_SKEW|DEFECTIVE',
    # The Delaporte's shifted-gamma mixing thickens the count tail enough for
    # 1.1e-4 of mass to run off the discrete grid, and the discrete update
    # path sizes its own grid (a caller or hints log2 is ignored), so the
    # deficit cannot be hinted away. The flag is the honest reading.
    'DelaporteMixedFrequency': 'DEFECTIVE',
    # --- Picking leaves the declared moments, by design -----------------
    # See test_layer_picks_reproduces_every_pick: the picks are reproduced
    # exactly, and the analytic severity moments are precisely what picking
    # decided to move away from. Read accuracy off the layer table instead.
    # ``LayerPicks.Compare`` is a portfolio holding one picked and one
    # unpicked unit, so it carries the picked unit's departure plus the skew
    # terms the pairing exposes.
    'LayerPicks.Uniform': 'SEV_MEAN|SEV_CV|AGG_MEAN|AGG_CV',
    'LayerPicks.Compare':
        'SEV_MEAN|SEV_CV|SEV_SKEW|AGG_MEAN|AGG_CV|AGG_SKEW',
    # The bottom-layer pick derives from the original expected loss (13500
    # became 14900), so the sum of loss by layer preserves the mean and the
    # two mean flags cleared; cv and skew remain what picking moved.
    'Capstone.SelectedLosses':
        'SEV_CV|SEV_SKEW|AGG_CV|AGG_SKEW',
    # --- Signed severity on a wrapped window ----------------------------
    # A signed window aliases at the ends; the numerics section exists to
    # show that behavior rather than to hide it. Three entries left this list
    # at 1.0.0a311, when the flag stopped being a ratio of two mean errors and
    # started measuring the convolution step
    # (``dev/done/plan-validation-punchup.md``); their residuals are 2.5e-10,
    # 1.1e-12 and 3.5e-7, and they were never aliasing.
    #
    # ``SignedPortfolioPair`` was the fourth and the only true positive left,
    # two Poisson(50) signed normals offset either way so the total mean was
    # zero, where wrap shows. The library reorganization rewrote it as two
    # one-claim fixed units with means 25 and 5, so the total is 30 and the
    # grid reproduces it: it no longer aliases and no longer belongs here.
    # THE SHIPPED LIBRARY NOW HAS NO ALIASING ENTRY AT ALL, which is worth a
    # deliberate decision rather than a silent gap.
    'WindowedContinuous': 'SEV_CV',
    'WindowedSimple': 'AGG_SKEW',
    # A signed reflection: the FFT skew of a reflected lognormal is the
    # hardest of the six moments to reproduce and the only one that misses.
    'SignedPremiumMinusLoss': 'AGG_SKEW',
    # --- Deliberately coarse or deliberately thick ----------------------
    # The cat model pins bs=2 on a billions-scale lognormal with cv 14.6, and
    # the thick portfolios are chosen to sit at the edge of what the grid can
    # carry. Both are the subject of their sections.
    'GrossCatXOL': 'SEV_MEAN|SEV_CV|SEV_SKEW|AGG_MEAN|AGG_CV|AGG_SKEW|INFEASIBLE',
    'RawLognorm': 'SEV_MEAN|SEV_CV|SEV_SKEW|AGG_MEAN|AGG_CV',
    'BodoffFour': 'SEV_MEAN|SEV_CV|AGG_MEAN',
    'PIRCatNonCatGross': 'SEV_SKEW',
    'PropertyCasualty': 'SEV_MEAN|AGG_MEAN',
    'ThickThickPortfolio': 'SEV_MEAN|SEV_CV|SEV_SKEW|AGG_MEAN|AGG_CV|INFEASIBLE',
    'ThickThinPortfolio': 'SEV_CV|SEV_SKEW|AGG_CV|AGG_SKEW',
}

#: Kinds that carry the ``valid`` / ``validation_explanation`` surface. A
#: ``sev`` or ``distortion`` is not a computed distribution with moments to
#: audit, and ``pnl`` / ``bvagg`` / ``xpnl`` report through their own surfaces.
VALIDATED_KINDS = ('agg', 'port')

FAST = [(k, n) for k, n in ENTRIES
        if n not in SLOW_ENTRIES and n not in CANNOT_BUILD]
SLOW = [(k, n) for k, n in ENTRIES
        if n in SLOW_ENTRIES and n not in CANNOT_BUILD]


def _check_built(kind, name, obj):
    """Assertions every built entry must satisfy, whatever its kind.

    The validation baseline is checked here, per entry, rather than in a pass
    of its own: a separate pass would have to rebuild the whole library a
    second time, which doubled the module's wall clock for no extra coverage.
    """
    assert obj is not None, f'{kind} {name}: build returned None'
    if kind not in VALIDATED_KINDS:
        return
    assert isinstance(obj.valid, Validation)
    assert isinstance(obj.validation_explanation, str)
    assert obj.validation_explanation, (
        f'{kind} {name}: valid is {obj.valid} with no explanation')
    assert isinstance(obj.summary_df, pd.DataFrame)
    assert not obj.summary_df.empty
    assert isinstance(obj.stats_df, pd.DataFrame)
    assert not obj.stats_df.empty

    # The baseline, asserted in both directions. An entry that starts failing
    # is a finding; so is one that starts passing, because a known-failure list
    # nobody checks for staleness stops being a baseline and becomes a comment.
    flags = str(obj.valid).replace('Validation.', '')
    if name in VALIDATION_BASELINE:
        assert flags == VALIDATION_BASELINE[name], (
            f'{name}: validation baseline is stale, expected '
            f'{VALIDATION_BASELINE[name]}, got {flags}. Edit '
            f'VALIDATION_BASELINE deliberately.')
    else:
        assert obj.valid.passes, (
            f'{name}: newly fails validation with {flags}. If that is correct '
            f'and deliberate, add it to VALIDATION_BASELINE with a reason.')


@pytest.mark.parametrize('kind, name', FAST, ids=[f'{k}:{n}' for k, n in FAST])
def test_every_library_entry_builds(library, kind, name):
    """Each entry builds and carries the surface its kind should carry."""
    _check_built(kind, name, library.build(name))


@pytest.mark.slow
@pytest.mark.parametrize('kind, name', SLOW, ids=[f'{k}:{n}' for k, n in SLOW])
def test_every_slow_library_entry_builds(library, kind, name):
    """Same, for the six entries that cost more than a second."""
    _check_built(kind, name, library.build(name))


@pytest.mark.parametrize('name, exc', sorted(CANNOT_BUILD.items()))
def test_entries_that_refuse_to_build_still_refuse(library, name, exc):
    """A deliberate refusal is part of the library's content.

    Pinned by exception *type name* rather than by class, because two of the
    three types are internal to the build path. What matters is that the entry
    raises, and raises the same way.
    """
    with pytest.raises(Exception) as info:                  # noqa: PT011
        library.build(name)
    assert type(info.value).__name__ == exc, (
        f'{name}: expected {exc}, got {type(info.value).__name__}')


def test_the_baselines_name_only_real_entries():
    """No dead names in either baseline.

    Cheap, and it is the failure mode a per-entry check cannot see: an entry
    renamed or deleted would silently drop out of ``VALIDATION_BASELINE``
    without anything noticing, leaving a stale line that reads as coverage.
    """
    names = {n for _k, n in ENTRIES}
    assert not (set(CANNOT_BUILD) - names), (
        f'CANNOT_BUILD names entries that are not in the library: '
        f'{sorted(set(CANNOT_BUILD) - names)}')
    assert not (set(VALIDATION_BASELINE) - names), (
        f'VALIDATION_BASELINE names entries that are not in the library: '
        f'{sorted(set(VALIDATION_BASELINE) - names)}')


def test_the_parametrization_is_not_empty():
    """Guard the guard: an empty entry list would pass everything vacuously."""
    assert len(ENTRIES) > 100
    assert len(FAST) > 100


# ----------------------------------------------------------------------
# The seven invariants, transcribed from the retired doc bodies
# ----------------------------------------------------------------------


def test_three_dice_is_exact():
    """Everything discrete lands on the grid, so the FFT is not an approximation.

    The sum of three dice is the one case where the transform is exact rather
    than merely accurate: the aggregate PMF is the exact threefold convolution
    of the die. That is what separates "is the method right" from "is the
    discretization fine", which every other entry has to hold apart.
    """
    a = build('DiceThreeEvenDice')
    # the die is dsev [2:12:2], the even faces 2..12, so E[X] = 7
    # E[A] = 3 x 7 = 21, and the estimate is EXACT, not merely close
    assert abs(a.actual_m - 21.0) < 1e-12
    assert abs(a.est_m - 21.0) < 1e-9
    # support runs 6..36; each extreme has probability (1/6)^3
    assert abs(a.density_df.loc[6, 'p_total'] - (1 / 6) ** 3) < 1e-12
    assert abs(a.density_df.loc[36, 'p_total'] - (1 / 6) ** 3) < 1e-12


def test_ph_distortion_is_a_concave_probability_map():
    """Concavity is what makes the induced premium coherent.

    A distortion maps probabilities to probabilities, so it fixes both ends;
    it must be increasing, or a likelier event would price as less likely; and
    it must be concave, which is what makes the premium subadditive so that
    combining two books never costs more than pricing them apart.
    """
    g = build('PHDistortion')
    # a distortion is a probability-to-probability map: g(0)=0, g(1)=1
    assert abs(g.g(0.0)) < 1e-12
    assert abs(g.g(1.0) - 1.0) < 1e-12
    # increasing, and CONCAVE (the coherence property)
    s = np.linspace(0, 1, 201)
    gs = g.g(s)
    assert np.all(np.diff(gs) >= -1e-12)
    assert np.all(np.diff(gs, 2) <= 1e-9)
    # a load, never a discount: g(s) >= s everywhere
    assert np.all(gs >= s - 1e-12)


@pytest.mark.slow
def test_layer_picks_reproduces_every_pick():
    """The picks clause hits the selected layer losses to floating point.

    Marked slow with the entry, which is one of the six over a second.

    The entry this pinned before the library reorganization was a mixed
    exponential whose tower stopped well below the end of the severity, so the
    mean was the picks *plus* an untouched tail and the tail had to be checked
    separately. ``LayerPicks.Uniform`` is a uniform on [0, 100] whose top pick
    is 100, so the tower covers the whole severity: the tail term is exactly
    zero and the mean is the picks and nothing else, which is a sharper
    statement of the same property.
    """
    tower = np.array([0., 50., 75., 100.])
    picks = np.array([30., 12., 5.])

    a = build('LayerPicks.Uniform')
    gross = build('agg Gross dfreq[1] sev 100 * uniform',
                  bs=a.bs, log2=a.log2)

    def layer_losses(obj):
        """Expected loss to each tower layer, off the limited expected value."""
        lev = obj.density_df['lev']
        return np.array([lev.loc[hi] - lev.loc[lo]
                         for lo, hi in zip(tower[:-1], tower[1:])])

    # every pick is reproduced, to floating point
    assert np.abs(layer_losses(a) / picks - 1).max() < 1e-8
    # picking moved the mean off the declared 50 and onto the picks
    assert abs(gross.est_sev_m - 50.) < 1e-9
    assert abs(a.est_sev_m - picks.sum()) < 1e-9
    # the tower covers the severity, so there is no loss above it to preserve
    assert layer_losses(gross).sum() == pytest.approx(gross.est_sev_m, rel=1e-9)


def test_limit_profile_derives_the_claim_count():
    """Premium and a loss ratio go in; the claim count comes out.

    The exposure clause solves for the count per band from expected loss and
    the severity limited at that band's limit, then adds the bands. The count
    is an output, which is the whole point of stating exposure this way.
    """
    a = build('ExposureLimitProfile')
    # the aggregate mean IS the premium-weighted expected loss, exactly:
    # 10000x0.8 + 20000x0.7 + 5000x0.6 + 5000x0.5 = 27500
    assert abs(a.actual_m - 27500.0) < 1e-9
    # and the FFT estimate agrees to grid accuracy
    assert abs(a.est_m / a.actual_m - 1) < 1e-6
    # E[A] = E[N] x E[X] still holds across the blended profile
    assert abs(a.n * a.sev_m - a.actual_m) < 1e-6


def test_occurrence_xol_cedes_without_changing_the_count():
    """Net plus ceded is gross, and an occurrence cover leaves the count alone.

    Both facts are read off ``Est EX``. The ``EX`` column carries the theoretic
    **gross** value in all three views, so comparing it across views compares a
    number to itself; the retired doc body's frequency check did exactly that
    and passed vacuously. The gross frequency has no ``Est EX`` at all, and
    correctly so: a gross claim count is an input, not an estimate.
    """
    a = build('OccurrenceXOL')
    est = a.reins_summary_df['Est EX'].xs('occ')
    # net + ceded = gross, on both severity and aggregate, to grid accuracy
    for component in ('sev', 'agg'):
        gross = est[('gross', component)]
        net = est[('net', component)]
        ceded = est[('ceded', component)]
        assert abs(net + ceded - gross) < 1e-6 * gross
    # occurrence cover changes what each claim costs, never how many there are
    assert np.isnan(est[('gross', 'freq')])
    assert abs(est[('net', 'freq')] - a.n) < 1e-9
    assert abs(est[('ceded', 'freq')] - a.n) < 1e-9


def test_neyman_inner_outer_matches_the_keyword():
    """An aggregate as the severity of another reproduces ``neymana`` exactly.

    A two level count is a stopped sum, so the inner/outer construction reaches
    any of them; the keyword exists only because the two level Poisson case is
    common enough to earn a closed form generating function. The atom at zero
    is load bearing: Neyman A counts the empty clusters too.
    """
    a = build('NeymanInnerOuter')
    builtin = build('agg NeymanBuiltIn 10 claims dsev [1] neymana 4',
                    bs=1, log2=10)
    # the two constructions agree to floating point across the whole support
    both = pd.concat([a.density_df.p_total, builtin.density_df.p_total], axis=1)
    assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-13
    # including the atom at zero, which is what keeping the empty clusters means
    assert abs(a.density_df.p_total.iloc[0]
               - np.exp(2.5 * (np.exp(-4) - 1))) < 1e-9
    # and on the moments the keyword computes analytically
    assert abs(a.est_m - 10.0) < 1e-9
    assert abs(a.est_m - builtin.est_m) < 1e-9
    assert abs(a.est_cv - builtin.est_cv) < 1e-9


def test_split_limit_policy_prices_the_per_accident_limit():
    """Two limits at two levels, and the inner one does all the work.

    A 100/300 auto policy caps each claimant at 100 and the accident at 300.
    With 1.25 claimants per accident an accident needs three claimants before
    the 300 can bind at all, so the per accident limit is worth almost nothing.
    """
    inner = build('SplitLimitClaimant')
    a = build('SplitLimitPolicy')
    # the frequency ! held the requested mean at 1.25 claimants, against the
    # 1.75 that truncating a Poisson(1.25) would otherwise give
    assert abs(inner.n - 1.25) < 1e-12
    # the outer mean is 100 policies times the per accident loss capped at 300
    capped = inner.density_df['lev'].loc[300.0]
    assert abs(a.est_m / (100 * capped) - 1) < 1e-9
    # the 300 per accident limit binds, but is worth under a hundredth of a
    # percent of expected loss
    saving = inner.est_m - capped
    assert 0 < saving < 1e-4 * inner.est_m
