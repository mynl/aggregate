The ``info`` string contract
============================

``Aggregate``, ``Portfolio`` and ``Distortion`` each expose a plain-text
``info`` property with a **fixed layout**: every row appears every time, in
the same order, for every instance of the class. There are no conditional
rows. A value that is not (yet) available — the object has not been
``update``-d, or the row does not apply to this object — renders as the
fixed placeholder ``n/a``. Two objects of one class therefore always emit
the same set of lines in the same order.

All three classes share one formatting convention, implemented by
:func:`aggregate.constants.info_row`: a label left-padded to 25 columns
(``INFO_LABEL_WIDTH``), no colon, value follows. The tail lines from
:func:`aggregate.tail.describe_lines` use the same width.

This contract was introduced in the hygiene-4 batch
(``dev/done/plan-hygiene-4.md``).


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
severity distribution        single component: ``long_name, support_description``;
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
P(loss)                      ``P(PnL < 0)`` off the signed density                     not a pnl, or not updated
validation_eps               moment-validation tolerance
reinsurance                  ``reins_kinds()``
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
  aggregate}`` (lower-cased ``reins_kinds()``).
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
