The ``info`` string contract
============================

Every **first-class class** — ``Aggregate``, ``Portfolio``,
``BivariateAggregate``, ``PnL``, ``Severity``, ``Frequency``, ``Bounds``,
``AllocationBounds``, ``PricingBounds`` — plus ``Distortion`` exposes a
plain-text ``info`` property with a **fixed layout**: every row appears every
time, in the same order, for every instance of the class. There are no
conditional rows. A value that is not (yet) available — the object has not been
``update``-d, or the row does not apply to this object — renders as the
fixed placeholder ``n/a``. Two objects of one class therefore always emit
the same set of lines in the same order.

All classes share one formatting convention, implemented by
:func:`aggregate.constants.info_row`: a label left-padded to 25 columns
(``INFO_LABEL_WIDTH``), no colon, value follows. The tail lines from
:func:`aggregate.tail.describe_lines` use the same width.

This contract was introduced in the hygiene-4 batch
(``dev/done/plan-hygiene-4.md``) and completed across the remaining
first-class classes at ``1.0.0a149`` (``[FCC-Surface-Sweep]``). The
executable half of the contract is ``tests/test_fcc_surface.py``.


Aggregate
---------

Rows in order. *Source* is the attribute or computation behind the value;
*n/a when* states the only condition under which the placeholder appears
(blank = always populated).

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
aggregate object name        ``self.name``
value_type                   configured label of the sign-convention role
                             (``_is_loss_value``)
claim count                  ``self.n``, format ``,.3f``
frequency distribution       ``self.frequency.freq_name``
severity distribution        single component: ``Severity.tail_description``;
                             else ``{k} components``
approximate                  method-of-moments marker; one row, no continuation
                             (fit detail lives in the note)
bs                           bucket size (``1/n`` form for ``bs < 1``)                 not updated
log2                         log2 of the number of buckets                             not updated
padding                      FFT zero-padding                                          not updated
sev_calc                     severity discretization scheme                            not updated
dsev_bucket                  discrete-atom placement scheme
normalize                    severity renormalization flag                             not updated
x_min                        realized output-window lower edge                         not updated
x_max                        realized output-window upper edge                         not updated
premium                      gross premium from ``stats_df ('meta', 'prem')``          no premium known
expected loss                empirical: ``est_m`` (loss) or
                             ``premium − est_m`` (pnl)                                 not updated
loss ratio                   expected loss / premium, format ``.1%``                   no premium or not updated
P(X=0)                       ``prob_eq_0`` — the break-even / no-loss atom
                             off the realised grid                                    not updated
validation_eps               moment-validation tolerance
reinsurance                  ``reins_kinds``
occurrence reinsurance       ``reins_description('occ')``
aggregate reinsurance        ``reins_description('agg')``
validation                   ``explain_validation()`` (single line)
frequency tail               ``tail.describe_lines`` row 1
severity tail                ``tail.describe_lines`` row 2
aggregate tail               ``tail.describe_lines`` row 3
bounded                      ``self.bounded``
id                           display-only 8-hex md5 of the canonical spec
===========================  =======================================================  =========================

Value enumerations:

- ``value_type`` ∈ the configured label pair — defaults ``{loss, payoff}``
  (``[labels]`` section of the config; see :mod:`aggregate.config`).
- ``approximate`` ∈ ``{exact, sgamma, slognorm}``.
- ``dsev_bucket`` ∈ ``{linear, nearest}``.
- ``normalize`` ∈ ``{True, False}``.
- ``sev_calc`` ∈ ``{discrete (= round), forward (= continuous), backward,
  moment}``.
- ``reinsurance`` ∈ ``{none, occurrence only, aggregate only, occurrence and
  aggregate}`` (lower-cased ``reins_kinds``).
- tail classes (each of the three tail rows) ∈ ``{bounded <
  super-exponential < exponential < subexponential < power-law}``, plus the
  ``unknown`` sentinel; frequency rows carry a log-concavity flag, severity /
  aggregate power-law rows carry the tail index ``alpha`` when known. The
  parenthetical names the frequency / severity family.
- ``validation`` ∈ ``{not unreasonable, n/a, not updated}`` or a
  comma-separated failure list (sev/agg mean, cv, skew, aliasing,
  reinsurance marker).


Portfolio
---------

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
portfolio object name        ``self.name``
value_type                   derived: the unanimous role of the units (mixed books
                             are rejected at construction)
aggregate objects            unit count ``len(self.line_names)``
allocation_method            natural-allocation method
bs                           bucket size                                               not updated
log2                         log2 of the number of buckets                             not updated
padding                      FFT zero-padding                                          not updated
sev_calc                     severity discretization scheme                            not updated
normalize                    severity renormalization flag                             not updated
x_min                        realized grid origin (``density_df.index[0]``)            not updated
x_max                        realized grid upper edge (last index + ``bs``)            not updated
premium                      gross premium from ``stats_df ('meta', 'prem')``          no premium known
expected loss                empirical: ``est_m`` (loss book) or
                             ``premium − est_m`` (payoff book)                         not updated
loss ratio                   expected loss / premium, format ``.1%``                   no premium or not updated
P(X=0)                       ``prob_eq_0`` — the break-even / no-loss atom
                             off the realised combined grid                            not updated
reinsurance                  ``reins_kinds`` — the per-unit look-through
                             (``none`` when no unit cedes)
aggregate tail               worst-of unit tail class, with driver unit(s)
bounded                      ``self.bounded`` (worst-of, certifiable)
last update                  ``np.datetime64`` of the last ``update``                  not updated
id                           ``hash_rep_at_last_update`` in hex (was labelled
                             ``hash``)                                                 not updated
===========================  =======================================================  =========================

Value enumerations:

- ``value_type``: as Aggregate (derived; empty portfolio defaults to the
  loss label).
- ``allocation_method`` ∈ ``{linear, lifted}``.
- ``aggregate tail``: the same tail-class ladder as Aggregate, suffixed
  ``(driver: <unit, ...>)`` naming the thickest unit(s).

``last update`` is the one remaining asymmetry with ``Aggregate`` —
``Aggregate``'s ``id`` is a display-only spec hash with no stored timestamp.


Bivariate (BivariateAggregate)
------------------------------

The joint bivariate aggregate (copula and ``netceded`` modes). Two axes are
always present, so the two per-axis blocks (``axis 0`` / ``axis 1``) always
emit the same rows in the same order. Grid rows render ``n/a`` before
:meth:`update`.

In ``netceded`` mode the keyword names the axis pair **x-then-y**: ``netceded``
→ (axis 0 ``net``, axis 1 ``ceded``); ``grossceded`` → (``gross``, ``ceded``);
``grossnet`` → (``gross``, ``net``).

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
bivariate object name        ``self.name``
mode                         ``copula`` or ``netceded``
components                   ``{axis 0 name} x {axis 1 name}``
copula                       ``repr(copula)``; ``comonotone (netceded)`` in
                             netceded mode
shared frequency             outer (shared) frequency name; the source agg's
                             frequency in netceded mode
claim count                  shared ``E[N]`` (source agg ``n`` in netceded),
                             format ``,.3f``
padding                      FFT zero-padding                                          not updated
axis 0 name                  component-0 name
axis 0 kind                  ``agg`` / ``pnl`` (copula) or the axis-0 view
                             ``net`` / ``gross`` (netceded)
axis 0 bs                    axis-0 bucket size (``1/n`` form for ``bs < 1``)          not updated
axis 0 log2                  axis-0 log2 grid length                                   not updated
axis 0 x_min                 axis-0 window lower edge                                  not updated
axis 0 x_max                 axis-0 window upper edge                                  not updated
axis 1 name                  component-1 name
axis 1 kind                  ``agg`` / ``pnl`` (copula) or the axis-1 view
                             ``ceded`` / ``net`` (netceded)
axis 1 bs                    axis-1 bucket size                                        not updated
axis 1 log2                  axis-1 log2 grid length                                   not updated
axis 1 x_min                 axis-1 window lower edge                                  not updated
axis 1 x_max                 axis-1 window upper edge                                  not updated
correlation                  realized output Pearson correlation                       not updated
copula tau                   copula Kendall ``tau``                                    netceded / no copula
tail deficit                 ``1 - sum(density)``                                      not updated
validation                   one-line :attr:`explain` headline (marginal mean +
                             deficit checks)                                           not updated
id                           display-only 8-hex hash of the structural fields
===========================  =======================================================  =========================

Value enumerations:

- ``mode`` ∈ ``{copula, netceded}``.
- ``axis k kind`` ∈ ``{agg, pnl}`` (copula) or ``{gross, ceded, net}``
  (netceded; the pair is set by the keyword — ``netceded`` / ``grossceded`` /
  ``grossnet``).
- ``validation`` ∈ ``{not unreasonable}`` or ``check: <marginal mean, tail
  deficit>`` -- the showpiece invariant being *each marginal reproduces its
  standalone aggregate* (full per-axis errors in :attr:`explain`).

The companion frames mirror the 1-D surfaces as **per-axis summaries** (the bv
*measures* its grid rather than running the 1-D method ladder): :attr:`explain`
(per-axis marginal-vs-standalone mean / cv error), :attr:`bs_window_df` /
:attr:`bs_description` (the realized per-axis grid), and :attr:`axis_support_df` /
:attr:`tail_description` (per-axis realized support + moments; the frame was
called ``tail_df`` until a171, which collided with the 1-D return-period table).
``describe`` and
``stats_df`` carry the per-component moment block (theoretical vs empirical) and
the joint dependence footer (correlation, copula tau).


Severity
--------

Terse, and **spec-only** — a ``Severity`` is never discretized on its own, so
there is no grid block and no ``est_*`` column. The moment rows are the
*achieved* (``actual_*``) values, which differ from the *declared*
``sev_mean`` / ``sev_cv`` whenever a layer, splice or shift applies.

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
severity object name         ``self.name``                                             no name given
severity distribution        ``self.long_name`` (the ``sev_name`` family)
declared mean                ``sev_mean`` as declared                                  not declared (0)
declared cv                  ``sev_cv`` as declared                                    not declared (0)
layer                        declared layer / atom support (``unlimited``,
                             ``{limit} xs {attachment}``, ``atoms [...]``,
                             ``signed, support [lo, hi]``)
conditional                  ``sev_conditional`` — layer moments conditional on
                             attaching
signed                       never-clamp (``ssev`` / negative-atom ``dsev``)
mean / cv / sd / skew        ``actual_m`` / ``actual_cv`` / ``actual_sd`` /
                             ``actual_skew`` — analytic, post-layer
severity tail                ``tail_description``
bounded                      ``self.bounded``
===========================  =======================================================  =========================


Frequency
---------

Terse. A parametric frequency is a *family* until the exposure fixes its mean,
so the moment rows render ``n/a`` until the owning ``Aggregate`` stamps ``en``.

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
frequency object name        ``self.name`` (== ``freq_name``)
frequency distribution       ``freq_name``
freq_a                       first shape parameter                                     unused / zero
freq_b                       second shape parameter                                    unused / zero
zero modified                ``freq_zm``
freq_p0                      modified ``P(N = 0)``                                     not zero-modified
E[N]                         first moment at the stamped ``en``                        ``en`` not set
SD(N)                        second central moment at ``en``                           ``en`` not set
var / mean                   dispersion vs Poisson (1 == Poisson)                      ``en`` not set
frequency tail               ``tail_description``
===========================  =======================================================  =========================


PnL
---

Terse. A ``PnL`` is a snapshot, so every row is available the moment it exists;
the moments are ``est_*`` (evaluated over the source's realised grid). The
narrative twin is ``construction_description`` / ``construction_explanation``.

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
pnl object name              ``self.name``
label                        ``self.label`` (resolved display label)
groups                       number of ledger groups
legs                         total declared legs across groups
role                         the single group's role, else ``multi-group``
result name                  ``result_name`` (the grand result row label)
E[consideration]             ``E_consideration`` — signed grand total
E[result]                    ``est_m``
SD(result)                   ``est_sd``
CV(result)                   ``est_cv``
skew(result)                 ``est_skew``
P(X=0)                       ``prob_eq_0`` — exact break-even probability
engine                       wrapped ``Aggregate`` / ``Portfolio``                     hand-built kernel P&L
source                       class of the stochastic source
economics                    ``resolved`` when a cession economics dict exists          plain P&L
===========================  =======================================================  =========================


Bounds
------

Terse and deliberately **cheap** — it reports the grid sizes rather than
materialising ``cloud_df``.

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
bounds object name           ``self.name`` (resolved from the source object)
kind                         fixed: ``pricing bounds (IME 2022)``
unit                         the priced unit (``total`` for a whole Portfolio)
E[X]                         ``tvar(0)`` of the resolved distribution
premium                      the target premium fixed at construction
margin                       ``premium - E[X]``
asset cap                    ``self.a``, or ``unlimited``
F(a)                         CDF at the asset cap (1 when unlimited)
p_star                       root of ``TVaR_p(min(X, a)) = premium``
n_p / n_s                    base p-grid and s-grid sizes
===========================  =======================================================  =========================


AllocationBounds / PricingBounds
--------------------------------

Both are ``_HullEngine`` subclasses, so the middle block
(``asset cap`` … ``s_floor``, from ``_hull_info_rows``) is **identical**; only
the identity rows above and the class-specific rows below differ.

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
allocation bounds object     AllB: ``self.port.name``
pricing bounds object        PrcB: ``{X} vs {risks}``
kind                         AllB: ``natural-allocation premium ranges by unit``;
                             PrcB: ``price ranges of risks consistent with the
                             reference``
reference risk               PrcB only: the X source's name
uniform reference            PrcB only: is X the uniform (Gini) reference
asset cap                    ``self.a``, or ``unlimited``
units / risks                the item names (``_item_label`` decides the label)
vertices                     rows in the exact piecewise-linear vertex table
premium range                the feasible ``[E[X], T_max]`` interval
s_floor                      deep-tail conditioning floor
additivity error             AllB only: ``max_p |Σ a_i(p) − T(p)|``
n_grid                       PrcB only: the sampling grid size
gini_level                   PrcB only: ``2P − 1``                                      X is not the uniform
===========================  =======================================================  =========================


Distortion
----------

===========================  =======================================================  =========================
Row                          Source / meaning                                          n/a when
===========================  =======================================================  =========================
distortion object name       ``self.name`` (display name if set, else kind)
kind                         kind abbreviation (registry key)
kind name                    spelled-out ``long_name``, lower case
shape                        value of the primary parameter (``param_name``)           kind has no ``param_name``
shape name                   the ``param_name`` string                                 kind has no ``param_name``
other params                 remaining ``decl_params`` as ``name=value``; multi-knot
                             / combo kinds render knot vectors / member names
                             compactly; ``none`` when empty
weights mean                 Kusuoka measure atom at ``p=0`` (mean-component
                             weight; was ``mu({0})``)
weights max                  Kusuoka measure atom at ``p=1`` (max / ess-sup
                             component weight; was ``mu({1})``)
interior atoms               boolean: does the Kusuoka measure have any atom in
                             the open interval (0, 1)
gini_p                       ``2∫g − 1``                                               NaN (multi-knot kinds)
area                         ``∫g = (gini_p + 1) / 2``                                 NaN (multi-knot kinds)
id                           ``self.id()`` — 8-hex machine-independent hash of the
                             structural fields
===========================  =======================================================  =========================

Kind registry (abbreviation → spelled-out name, with each kind's primary
``shape name`` and ``other params``):

===========  ===========================  ============  =====================
kind         kind name                    shape name    other params
===========  ===========================  ============  =====================
ph           Proportional Hazard          a             none
wang         Wang-normal                  lam           none
dual         Dual Moment                  b             none
tvar         Tail VaR                     p             none
ccoc         Constant CoC                 —             r
bitvar       BiTVaR                       —             p0, p1, w1
wtdtvar      Weighted TVaR                —             ps=[...], wts=[...]
minimum      Minimum                      —             members=[...]
mixture      Mixture                      —             members=[...], wts=[...]
beta         Beta                         —             a, b
power        Power                        —             x0, x1, alpha
cll          Capped Loglinear             —             b
clin         Capped Linear                —             slope
lep          Leverage Equivalent Pricing  —             r
ly           Linear Yield                 —             r
===========  ===========================  ============  =====================

Dropped rows (vs the pre-hygiene-4 format): ``display`` (the display name is
already the object name), ``strict-pricing``, and there is no update
timestamp — a distortion is intrinsic, it has no ``update``.
