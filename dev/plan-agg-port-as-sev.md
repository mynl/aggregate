# [Agg-As-Severity] Plan: agg and port references as severities in DecL

Status: DRAFT for author review, revised 2026-08-16 (fourth round: granular
implementation guidance for the tail descriptor, section 4.6, and the
commensurability step, section 4.7, which now lands as its own bump, phase
B2). Planning only; nothing has been implemented. File and line references
verified against the working tree at 1.0.0a289.

## 1. Motivation

Allow an aggregate (or portfolio) already present in the Underwriter recipe base
to serve as the severity of another aggregate. The motivating case is a US
personal auto split limit, a 100/300 policy: 100 per claimant, 300 in the
aggregate. In the author's formulation:

```
agg SL 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt ! hints{log2=16; bs=1/32}
agg Auto.SL 5000 claims 300 xs 0 sev agg.SL mixed gamma .2
```

`SL` is the per policy claims distribution: per claimant limit 100 on its layers
clause, at least one claim per policy through the zero truncated Poisson (the
freq clause `!` keeps the requested mean of 1.5). `Auto.SL` compounds 5000
policies, applying the 300 aggregate cap as an ordinary occurrence limit on the
reference severity. Because the zt inner has no mass at zero, no severity side
`!` is needed here; section 5 covers the zero mass variants. "In the recipe
base" means anything defined earlier in the session or loaded from a library.
Scope is references only (`sev agg.NAME`, `sev port.NAME`, `ssev` variants); an
inline full DecL program in severity position is out of scope. Depth 1 must
work; deeper nesting falls out naturally with a cycle guard.

## 2. Conceptual model: a named reference to a certified, fully formed dsev

The governing mental model (author, 2026-08-16, terminology settled in the
third round): the program **parses with the reference and handles the
reference subsequently**. It is not textual injection (the inner is fully
parsed on its own; no spliced text, so pathologies like a doubled `!` cannot
arise), and that is a key reason the language only allows calling inners by
name (`agg.NAME`), never inline. It is also not object sharing: the inner may
exist only as a recipe and never have been built, for example an entry in
`library.agg`, a likely scenario. The user takes a declaration they have
created and evaluated and uses it as a severity; they are certifying they are
happy with it. Consequences, all load bearing:

1. **The reference is treated as a fully formed dsev.** The inner's output pmf
   becomes a discrete severity, exactly as though the user had transcribed
   `dsev [xs] [ps]` by hand, and it is then quantized onto the outer grid the
   same way any dsev is. No blending, no smoothing, no continuous reading.
2. **The query for the dsev representation is nullary.** No arguments are
   passed, just as no arguments are passed to instantiate a lognorm. The inner
   knows how to recompute itself exactly from its own declaration and hints.
   The outer's update arguments play no role in how the inner is computed or
   materialized (in particular `normalize` is the inner's choice, via its own
   hints).
3. **Hygiene rule, firm: a referenced inner must carry explicit `log2` and
   `bs` hints.** (Author ruling 2026-08-16.) With `hints{log2=...; bs=...}` on
   the inner declaration, the dsev representation is pinned entirely by the
   recipe base, independent of ambient underwriter defaults, and the outer's
   bucket selection can read the inner's bs without building anything. A
   reference to an inner without both hints raises a clear error at resolution
   time. The certification path is `with_hints` (section 4.10): get the inner
   right interactively, then `build(inner.with_hints())` re registers it with
   its resolution pinned; the error message names the method and also shows
   the exact `hints{log2=...; bs=...}` text to paste.
4. **Its dsev moments are its theoretical moments.** When the outer asks the
   severity for moments, it gets the exact discrete moments of the materialized
   atoms, reported as theoretical (they feed `stats_df['mixed']` and the bucket
   sizer). That is what the object outputs, so that is the truth the outer
   works from.
5. **The one exception is tail reporting** (section 4.6): the outer reports the
   inner's theoretical tail character, not the dsev's. `agg UB dfreq [1] sev
   gamma 100 cv 1` used as an inner reports an unbounded right tail even though
   the materialized dsev is bounded at its largest atom. Reporting only;
   numerics stay dsev.
6. **The only update interaction is bucket awareness.** The outer's bucket
   selection becomes aware of the inner's bs choice (section 4.7). There is no
   other grid coupling and no fancy joint optimization.
7. **Certification rides the declaration plus hints, not a cached object.**
   There is no built object cache in the recipe base, so resolution rebuilds
   the inner from its recipe. With the mandatory hints the rebuild is fully
   deterministic, so the object the user evaluated is reproduced exactly. A
   literal reuse cache is the `[Agg-As-Severity-Result-Cache]` follow up.

## 3. Current state (verified findings)

1. **The Python machinery half exists, is broken in practice, and is being
   replaced rather than repaired.** `SeverityMeta`
   (`src/aggregate/_severity.py:2233`, `sev_kind = 'meta'`) survives from 0.x:
   `_classify_sev` (`_severity.py:607`) dispatches an `Aggregate` or
   `Portfolio` instance passed as `sev_name` to it, with the `sev_a` and
   `sev_b` spec slots repurposed as `log2` and `bs`. Known defects: the
   Aggregate branch calls the nonexistent `easy_update` (stale docstring trace
   at `_aggregate.py:3561`); a NaN `sev_a` default passes the `if log2`
   truthiness test; and the author reproduced a live failure on 2026-08-16,
   `build('port pSL agg SL 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt
   !').as_severity()` raises `IndexError: string index out of range`, a string
   indexing of the object valued `sev_name` somewhere before the known broken
   branch is even reached (exact site diagnosed in phase A; the repro becomes
   a phase A test). It also updates the source in place when
   resolutions differ (flagged for review at the refactor) and builds a
   continuous `ss.rv_histogram` hybrid with no `support_atoms`, no exact
   discrete moments, no lattice. Untested, undocumented. Author ruling: the
   idea is useful and stays, rebuilt afresh on the dhistogram foundation
   (section 4.9), working on **both** aggs and ports; the `meta.` keyword is
   never reinstated.
2. **The DecL surface is cheaply constructible.** The `BUILTIN_AGG` and
   `BUILTIN_PORT` terminals exist (`decl.lark:796` region), the `ID` terminal's
   negative lookahead already excludes the `agg.` and `port.` prefixes
   (`decl.lark:848`), and all four dotted terminals are already covered by
   `parser_errors._TERMINAL_LABELS` (`parser_errors.py:161`), the Pygments
   lexer, and `agg.sublime-syntax`. **No new terminal, no keyword mirror edits,
   and no app side `decl-keywords.json` ripple**: the change adds no keywords.
3. **Every existing DecL reference inlines the referenced spec eagerly at parse
   time** (`Underwriter._safe_lookup`, `underwriter.py:1784`, returns a
   deepcopy of the stored spec dict). That cannot work here: the severity is
   the inner's computed output (`xs`, `agg_density`), which exists only after
   an update. This is DecL's first deferred, build time reference. Precedent
   for carrying a symbolic marker on a spec that the unparser renders back:
   `_engine_port_spec` and `_engine_port` (`parser.py:746`). The
   `[Unparser-Reference-Gaps]` item in `dev/TODO.md` already proposes a
   `sev_ref` spec key; this design adopts that name and establishes the
   pattern.
4. **There is no built object cache.** Recipes always store `object = None`;
   `_factory` sets objects only on `replace()` copies. Update orchestration
   lives in `build_many` after `_factory` has returned, and
   `Aggregate.__init__` constructs its `Severity` components immediately, so
   inner resolution must happen at the top of `_factory`
   (`underwriter.py:1087`). The comment at `underwriter.py:1107` records that
   `parsed.spec` is the stored recipe's dict, so the resolver must return a new
   dict, never mutate in place.
5. **A dhistogram is the right materialization.** Building the severity as
   `sev_name = 'dhistogram'` from the inner's `(xs, agg_density)` (for a port,
   `(density_df.loss, density_df.p_total)`) lands in `SeverityDHistogram`
   (`_severity.py:2133`): exact first three moments from the atoms,
   `support_atoms` for lattice detection, `_DiscreteRV` with exact
   `layer_moments`, and the mean preserving linear rebucket onto the outer grid
   (`_aggregate_compute.py:299`, `Aggregate._rebucket_to_grid`). It is
   literally the fully formed dsev of the conceptual model.
6. **The moment circularity dissolves.** The inner sizes and computes itself
   from its own declaration (hints), and then the outer's severity moments are
   the exact moments of the inner's output pmf, available to
   `MomentAggregator` before the outer update. Under reinsurance the inner's
   theoretical moments are gross only and the post reinsurance law exists only
   after the inner update, which is precisely why resolution must build and
   update the inner first.
7. **Reinsurance semantics come free where reinsurance is used.** After
   `aggregate ceded to 300 xs 0`, `agg_density` is the ceded view
   (`_reinsurance.py:304` region), including the atom at 300. "The
   distribution the object outputs" is already the right array. (Section 5
   explains why the docs nonetheless steer users toward occurrence limits when
   they suffice.)
8. **Mass at zero, when the inner has it, is part of the law.** A plain
   Poisson 1.5 inner has P(0) of about 0.22; a zt inner has none. With no
   layers clause on the outer the zero atom is kept (`exp_attachment = None`);
   a layers clause conditions on exceeding the attachment by default, exactly
   as for `dsev` today, and the severity side `!` keeps the layer
   unconditional. Section 5 works through the consequences.
9. **A per declaration sizing channel already exists.** The `hints{...}` allow
   list `_HINT_KEYS` (`underwriter.py:173`) includes `log2`, `bs`, `padding`,
   `normalize`, `bucket_sizing_p`, `sev_calc`, `discretization_calc`,
   `force_severity`, `x_min`, `x_max`, resolved caller wins by
   `_resolve_hints`. The hygiene rule of section 2 rides on it.
10. **Single program, define then use, works by construction.**
    `_interpret_program` parses statements in order and calls `add_recipe`
    after each parse, so within one `build` or `build_many` call a later
    statement's parse time existence check sees an earlier statement's recipe,
    and the build time resolution then rebuilds it from that recipe. A two
    statement program defining `SL` and using it in `Auto.SL` works in one
    call (pinned by test; the inner is built twice in that program, once as
    its own output object and once during resolution, which the result cache
    follow up would eliminate).

## 4. Design

### 4.1 Syntax and grammar

`sev agg.NAME`, `sev port.NAME`, `ssev agg.NAME`, `ssev port.NAME`, each with an
optional unconditional marker `!` and an optional `as` label. The keyword is
required, and the reference is a terminal leaf: it cannot take picks, weights,
scaling, shifting, or splices (their semantics on a compound output are
undecided). Sketch of the grammar diff in `src/aggregate/decl.lark` (current
`sev_clause` block at lines 467 to 470); the exact factoring is finalized at
implementation with the ambiguity sweep as arbiter:

```
sev_clause: SEV sev as_label            -> sev_clause_sev
          | SSEV sev as_label           -> sev_clause_ssev
          | dsev as_label               -> sev_clause_dsev
          | BUILTIN_SEV as_label        -> sev_clause_builtin
          | SEV sev_ref_leaf as_label   -> sev_clause_ref
          | SSEV sev_ref_leaf as_label  -> sev_clause_ref_signed

sev_ref_leaf: BUILTIN_AGG               -> sev_ref_agg
            | BUILTIN_PORT              -> sev_ref_port
            | sev_ref_leaf "!"          -> sev_ref_uncond
```

Ambiguity analysis: `sev` can never derive a `BUILTIN_AGG` or `BUILTIN_PORT`
token (the `ID` lookahead excludes the prefixes and `sev`'s only dotted leaf is
`BUILTIN_SEV`), so the new alternatives are token disjoint from `SEV sev` and
`SSEV sev`. Requiring the keyword removes any interaction with the bare
`builtin_agg` rename form of `agg_body`. The optional `as` label rides the
existing interior label machinery. Acceptance check:
`tests/test_grammar_ambiguity.py` must report `KNOWN_AMBIGUOUS` unchanged.

**The two `!`s** (author, 2026-08-16, and a docs must): the severity side `!`
here means an unconditional layer, as it does on `sev` and `dsev` today. It is
unrelated to the frequency side `!` that modifies `zt`/`zm` to keep the
requested mean. A reference program can legitimately carry both, in different
clauses of different statements; the docs and the error messages must keep them
straight.

Layers clause semantics: `agg X 5000 claims 300 xs 0 sev agg.SL2 ! poisson`
applies `exp_limit = 300`, `exp_attachment = 0` to the materialized dhistogram
through the ordinary `exp_*` keys, and the `!` keeps the zero atom under the
layer. Without `!` the layers clause conditions on exceeding the attachment,
exactly as today (inert when the inner has no zero mass, consequential when it
does; section 5). Unlayered references take the mean preserving linear scatter;
layered references fall to the sf difference path, the same behavior `dsev`
has.

### 4.2 Signed rules (`sev` versus `ssev` on a reference)

The usual combinations are `sev` over a nonnegative inner and `ssev` over a
signed inner. The full matrix (author, 2026-08-16):

| Outer keyword | Inner support | Behavior |
|---|---|---|
| `sev` | nonnegative | The usual case. No warning. |
| `ssev` | signed | The usual signed case. Negative atoms kept; the severity auto signs through the negative atom rule in `SeverityDHistogram`. No warning. |
| `sev` | signed | Works, with a WARNING: negative atoms are collapsed into the zero atom (the clamp at 0 convention, matching `sev 100 - lognorm` versus `ssev 100 - lognorm`), and the combination usually signals a value convention mismatch (a P&L style inner consumed in a loss context). |
| `ssev` | nonnegative | Allowed silently; harmless. |

The clamp happens in the materialization step (section 4.5): mass at negative
atoms moves to the zero atom before the dhistogram is formed. The warning
category is vetted at implementation (plain `UserWarning` unless a house
category fits better).

### 4.3 Spec representation and transformer

New spec key **`sev_ref`**, holding the verbatim dotted id string (`'agg.SL'`,
`'port.PBook'`). The recipe base spec carries only this symbolic reference plus
the ordinary `sev_signed` and `sev_conditional` keys contributed by the `ssev`
and `!` forms, which keeps it re resolvable against the current recipe base and
trivially renderable. The built object's `_spec` carries the resolved
dhistogram arrays, so `Aggregate(**a.spec)` reconstitutes the frozen resolved
object standalone, while `a.program` preserves the reference form for a re
resolving rebuild.

Transformer handlers in `parser.py`, next to `sev_clause_builtin` (line 1732):
the `sev_ref_*` leaf handlers return `{"sev_ref": ref}` (plus
`sev_conditional = False` for the `!` form), after a kind checked existence
check via `self.safe_lookup(ref)` whose returned spec copy is discarded
(resolution is deferred to build time). The clause handlers attach
`_severity_label` and, for `sev_clause_ref_signed`, `sev_signed = True`.

Guards in `builtin_agg_homog`, `builtin_agg_plus`, and `builtin_agg_minus`
raise `ValueError` when the referenced spec carries `sev_ref` (the algebra
scales `sev_mean`, `sev_scale`, `sev_loc`, none of which exist on a deferred
reference); `builtin_agg_inhomog` is frequency only and stays legal.

`sev_ref` is deliberately not an `Aggregate.__init__` kwarg: the underwriter
pops it before construction. A user calling `Aggregate(**spec)` on a raw recipe
spec gets a truthful `TypeError`; the docs note that references are a build
level feature.

Unparser: an early branch in `_render_sev_clause` (`decl_writer.py:295`)
renders `sev agg.SL`, `ssev agg.SL`, and the `!` suffix from `sev_ref`,
`sev_signed`, and `sev_conditional`. The `_is_dsev` discriminator is untouched
(a `sev_ref` spec has neither `sev_name` nor `sev_xs` in the recipe base). This
makes agg and port references the first dotted references that round trip as
references, the pattern `[Unparser-Reference-Gaps]` asks for.

### 4.4 Build time resolution in the Underwriter

Hook at the top of `_factory` (`underwriter.py:1087`), immediately after
unpacking:

```python
if kind in ('agg', 'pnl', 'xpnl') and 'sev_ref' in spec:
    spec = self._resolve_sev_ref(name, spec)
```

`pnl` and `xpnl` are included because the parser merges an agg engine's spec
into the pnl spec, so an engine using `sev agg.X` carries `sev_ref` on the pnl
spec. Portfolio unit specs are phase C.

New method `_resolve_sev_ref(self, name, spec)`, joining the `_resolve_hints`
and `_resolve_reins_economics` family. It returns a new dict and never mutates
its argument. Algorithm (the docstring Notes section carries it):

1. Split `spec['sev_ref']` into `(rkind, rname)` exactly as `_safe_lookup`
   does.
2. Cycle guard: `self._sev_ref_stack`, a list of `(kind, name)` tuples
   initialized in `__init__`. If `(rkind, rname)` is already on the stack,
   raise `ValueError` naming the full chain (`severity reference cycle:
   agg.A -> agg.B -> agg.A`). Push, then try/finally pop. This also catches
   the self reference that redefinition makes reachable.
3. Look up the current stored recipe as a `replace()` copy; a `KeyError`
   becomes a clear `ValueError` (the referenced entry was removed after
   parsing).
4. **Hygiene check** (section 2 item 3, firm): the inner recipe's hints must
   supply both `log2` and `bs`. If not, raise a clear `ValueError` whose
   message names `with_hints` (section 4.10) and includes ready to paste
   `hints{log2=...; bs=...}` text computed from the inner's own bucket window.
5. `rec = self._factory(rec)`. The recursion makes depth N fall out with no
   extra code; the stack guard bounds it. Each nesting level is subject to the
   same hygiene check for its own references.
6. Update with the inner's own hints: `rec.object.update(log2=hint_log2,
   bs=hint_bs, ..., force_severity=True, **kw)`, the Portfolio variant adding
   `remove_fuzz=True` to match `build_many`. The query is nullary in the
   conceptual model's sense: nothing flows in from the outer; the inner
   recomputes itself exactly as declared.
7. Materialize through `_dhistogram_from_object` (section 4.5), passing the
   inner's effective `normalize` and the outer's signed context, and capture
   the inner's tail descriptor (section 4.6). The inner's bs is recorded for
   the outer's bucket awareness (section 4.7).
8. Return the spec with `sev_ref` removed and `sev_name = 'dhistogram'`,
   `sev_xs`, `sev_ps` added; log at INFO with the inner id, log2, bs, atom
   count, and mean.

Iteration policy, stated and pinned by test: every build of the outer re
resolves the inner fresh against the current recipe base. With mandatory hints
this is exactly reproducible, correct under last write wins redefinition, and
holds no hidden state; the cost is one inner FFT update per outer build,
milliseconds to tens of milliseconds at typical resolutions. Caching a built
object on the stored recipe is explicitly deferred; recorded as the follow up
`[Agg-As-Severity-Result-Cache]`, to be taken up only if profiling ever hurts.

### 4.5 Materialization as a discrete severity

Shared function `_dhistogram_from_object(obj, normalize, signed)` living in
`_severity.py` (module level, so the rebuilt `SeverityMeta` of section 4.9 and
the underwriter resolver use one implementation), returning `(xs, ps)`:

- `Aggregate`: `xs = obj.xs`, `ps = obj.agg_density`, the post reinsurance
  output view when reinsurance is present.
- `Portfolio`: `xs = obj.density_df.loss.values`, `ps =
  obj.density_df.p_total.values`.
- Guard: raise a clear `ValueError` if the object has never computed itself (a
  nullary query cannot be answered by an object with no density).
- Cleanup: `ps = np.maximum(ps, 0)` (FFT fuzz can leave tiny negatives, and
  `validate_discrete_distribution` neither checks nor renormalizes
  probabilities), then drop `ps == 0` atoms, which typically shrinks
  `2**log2` entries to the mass carrying few thousand.
- Signed handling per section 4.2: in an unsigned outer context, mass at
  negative atoms moves to the zero atom with a warning; in a signed context
  the atoms pass through.
- Renormalization honors the inner's effective `normalize` (the value resolved
  from the inner declaration's hints, default True): True renormalizes the
  materialized `ps`; `hints{normalize=False}` on the inner keeps the deficit
  and the outer compound is faithfully defective. Either way, warn when the
  deficit exceeds about 1e-6, which signals an undersized inner window (raise
  the inner `hints{log2=...}`).
- Zero mass, when the inner has it, is kept: it is part of the distribution
  the object outputs.

### 4.6 Tail reporting (the one exception to pure dsev semantics)

Author requirement: the outer reports the inner's theoretical tail as the
tail. `agg UB dfreq [1] sev gamma 100 cv 1` used as an inner has a
materialized dsev bounded at its largest atom, but the outer's tail reporting
must describe the right tail as unbounded, because the theoretical object is.

Design: `_resolve_sev_ref` captures a small tail descriptor from the inner at
resolution time. The descriptor is a read only attribute on the materialized
severity (name vetted before coding), minimally the right tail character,
`'unbounded'` or `'bounded'` with the finite bound when bounded. Numerics are
untouched everywhere: bucket sizing, moments, and the FFT all use the dsev.

**Derivation procedure** (run inside `_resolve_sev_ref` on the just updated
inner; implement exactly this order):

1. Severity side, per component of the inner. **Descriptor precedence is
   absolute**: if a component itself carries a tail descriptor (it is a
   materialized reference, the depth 2 case), use that descriptor and never
   read the component's `detachment` or largest atom, which are finite for
   every materialized dsev and would silently break transitivity. Otherwise
   the component is bounded iff its theoretical detachment is finite (an
   explicit layers limit, or an inherently bounded family); an unbounded
   scipy family with no limit is unbounded.
2. Frequency side: the inner's frequency is bounded iff its support is finite
   (`dfreq`, `fixed`, binomial); Poisson, negbin, and the mixed families are
   unbounded, and zero truncation or modification does not change
   boundedness. Key this off the frequency kind through a small helper or
   lookup (name vetted; do not infer numerically).
3. Subject aggregate: unbounded to the right iff (some severity component is
   unbounded and P(N at least 1) is positive) or (the frequency is unbounded
   and the severity has positive mass above zero). Note the second clause: a
   bounded severity under an unbounded frequency still gives an unbounded
   aggregate.
4. Reinsurance transformation to the output view. Occurrence covers first:
   a `ceded to` occurrence view with finite limit bounds each claim, then
   step 3 reruns with the transformed per claim tail (bounded per claim with
   bounded frequency gives bounded; with unbounded frequency, unbounded).
   Then the aggregate cover: a `ceded to` aggregate view with finite limit is
   bounded at that limit; a `net of` view is unbounded iff the subject is (a
   finite cession cannot bound an unbounded subject).
5. The outer's own layers clause overrides at the end: a finite `exp_limit`
   applied to the reference bounds the reported severity tail at that limit
   regardless of the descriptor; the descriptor survives only under an
   infinite outer limit.

Consumers pinned for v1: `tail_behavior_df` only, plus the validation
narrative only if the wiring is trivial. Nothing else reads the descriptor,
and nothing writes it after resolution.

On the author's question whether the tail drives bucket selection: it does,
and the model routes that influence through the inner. The inner's theoretical
tail drove the inner's own window choice (its `bucket_sizing_p` percentile,
now pinned by its mandatory hints), and the certified dsev encodes that
choice; the outer then sizes from the dsev exactly as it would for a hand
written one, including legitimately treating it as bounded. The numeric
safeguard for tail mass beyond the inner window is the deficit warning of
section 4.5 (and raising the inner `hints{log2=...}`), not the descriptor,
which stays reporting only.

### 4.7 Commensurable bucket sizes (author requirement, simplified)

The inner and outer bucket sizes must never be incommensurable; ideally the
inner bs divides the outer bs. Per the author's direction this is a **final
query step at the end of bucket selection**, not an optimization woven through
the sizer. It lands as its own bump, phase B2
`[Agg-As-Severity-Commensurable-Grids]`, immediately after the feature bump
(decided at review: the default, not a contingency), so the
`_bucket_window.py` diff stays isolated and readable.

Implementation procedure (implement exactly this shape):

- **Carrier.** The materialized reference severity carries its inner grid
  spacing `d` as a float attribute set by `_resolve_sev_ref` (name vetted;
  `None` on every other severity kind). Never infer `d` from `np.diff` of the
  atoms; it is known exactly on the resolution path, and the hygiene rule
  even makes it statically visible in the inner's declaration.
- **Hook.** One new module level function in `_bucket_window.py` (name
  vetted), called once at the end of `bs_window` after the winning candidate
  row is selected and before results are returned. No changes to `_size`, to
  the candidate row construction, or to the priority logic.
- **Decision table**, with `b0` the selected bs and integer tests via
  `math.isclose` at an explicit relative tolerance (1e-9):
  - no component exposes `d`: return unchanged (the overwhelmingly common
    path, zero cost);
  - `d / b0` is an integer (the outer grid is finer; inner atoms land exactly
    on grid points): no change. This also covers an exact discrete winner
    (bs 1) over an integer lattice inner;
  - `b0 / d` is an integer: already commensurable, no change (do not force a
    power of two when the estimator or the user landed on an exact integer
    multiple);
  - otherwise, auto sized bs: snap to `d * 2**m` with
    `m = max(0, ceil(log2(b0 / d)))`, never below `d` (the outer severity is
    the inner's output, so an estimate below `d` snaps to `d` itself). Then
    **re invoke `_size` for the winning row's `(x_lo, x_hi)` passing the
    snapped value as `lattice_bs`**, so origin, span, and the log2 need all
    come from the one existing kernel; never duplicate window arithmetic at
    the call site. `d * 2**m` is exact in binary floating point, so the
    snapped grid carries no representation fuzz;
  - otherwise, explicitly pinned bs: leave it (no back doors) and warn,
    naming both values and the nearest commensurable alternative on each side
    (the adjacent integer multiples of `d`).
- **Reporting.** The adjustment appends to `bs_window_df` and
  `bs_explanation` with the inner's id, `d`, and the pre and post snap bs.
- **Multiplicity.** The v1 grammar admits at most one reference per
  aggregate, so at most one `d` reaches a sizing. The function nonetheless
  accepts a set; if it ever sees more than one distinct `d`, it snaps to the
  largest and warns (future proofing for port unit sizing, phase C).
- **Alignment.** Outer origins are multiples of the outer bs, hence of `d`,
  so inner atoms stay on the `d` sublattice; signed inners are covered by the
  existing origin flooring in `_size`.

### 4.8 Sizing control and iteration

No new `build` or `build_many` kwargs (author ruling): the inner declaration's
`hints{log2=...; bs=...}` clause is the sizing channel, now mandatory for
referenced inners (section 2 item 3), DecL native, persistent in the recipe
base, honored at every nesting depth because each level reads its own hints
during its own resolution.

Accuracy story, to be stated in the docs: the linear rebucket preserves the
materialized severity mean exactly; the outer's theoretical moments are exact
given the inner grid; with commensurable grids the scatter weights are exact
binary fractions; the residual error is the inner discretization error (higher
moments) plus outer grid scatter (higher moments). Raise the inner
`hints{log2=...}` for thick tails or when the outer's validation flags fire.

### 4.9 SeverityMeta afresh, with an Aggregate.as_severity twin

Author rulings: start afresh (no repair of the hybrid), never reinstate the
`meta.` keyword, and **keep `as_severity`, working on both ports and aggs**
(the idea is useful; the author reproduced the current breakage on
2026-08-16). The `_classify_sev` dispatch (an `Aggregate` or `Portfolio`
instance passed as `sev_name`) survives as the programmatic API;
`SeverityMeta._build` is reimplemented:

1. The programmatic path answers the nullary query from the object's current
   computed state. An object that has never computed itself raises a clear
   `ValueError` telling the user to update it first. **The in place update
   side effect is removed** (flagged for review at the refactor; this is the
   review outcome). The `easy_update` call disappears with it, and the stale
   alias claim in the `update` docstring (`_aggregate.py:3561`) is deleted.
2. The `sev_a` and `sev_b` repurposing as log2 and bs is retired. If either is
   passed with a value conflicting with the source's current grid, raise a
   clear `ValueError` (update the source at the resolution you want, then
   convert). `Portfolio.as_severity` stops passing them.
3. `_build` routes through `_dhistogram_from_object` (section 4.5) and the
   `SeverityDHistogram` machinery: `_DiscreteRV`, exact moments,
   `support_atoms`. The `rv_histogram` hybrid (the `bs*1e-7` spike
   construction) is deleted.
4. **New `Aggregate.as_severity`**, mirroring `Portfolio.as_severity`'s
   signature (`limit`, `attachment`, `conditional`), so the programmatic
   surface is symmetric across the two kinds. The name is clean on `Aggregate`
   (rg verified: `as_severity` exists only on `Portfolio` today); final vet at
   implementation per the house rule.
5. The class name `SeverityMeta` and `sev_kind = 'meta'` remain internal; **no
   `meta.` keyword is ever reinstated in DecL.**

The author's 2026-08-16 repro,
`build('port pSL agg SL 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt
!').as_severity()`, becomes a phase A regression test, alongside the same
shape on a bare agg via the new twin.

### 4.10 `with_hints`: the certification helper

New public method on both `Aggregate` and `Portfolio` (author, 2026-08-16):
`with_hints(**extra)` returns the object's DecL program with a `hints{...}`
clause carrying `log2`, `bs`, and `normalize` from the object's current
computed state, merged with any extra hints passed as keyword arguments (any
`_HINT_KEYS` member). It requires the object to have been updated (the values
otherwise do not exist) and replaces an existing hints clause rather than
duplicating one. The SUVA path is: get the inner right interactively, then
`build(inner.with_hints())`, which re registers the declaration with its
resolution pinned, morphing a potential inner into a certified one. The
resolver's hygiene error names this method.

## 5. Preferred modeling idiom, the two `!`s, and the zt identity

Docs guidance (author, 2026-08-16): express the inner and outer **as simply as
possible**. The preference order for achieving a given structure is: no
limits, then exposure (occurrence) limits, then aggregate reinsurance, then
occurrence reinsurance, then both. The reinsurance machinery is substantial
and should be avoided when plain limits reach the same law. The docs already
carry a substantial discussion of reinsurance versus exposure occurrence
limits; the new guidance links to it rather than restating it.

For the split limit the recommended form is the headline of section 1: a zt
inner (no mass at zero, freq `!` keeping the requested mean) and the 300 cap
as an occurrence limit on the outer, with no severity side `!` needed because
there is no zero atom to protect.

The discussion below uses three inners, identical except for the frequency
clause (author clarification 2026-08-16):

```
agg SL   1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt !   # headline: truncated, mean held at 1.5
agg SLzt 1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson zt     # truncated, parameter 1.5, mean about 1.93
agg SL2  1.5 claims 100 xs 0 sev gamma 50 cv 2 poisson        # plain Poisson, P(0) about 0.22
```

(each carrying its `hints{log2=...; bs=...}` per the hygiene rule, omitted here
for readability).

When the inner does have mass at zero (`SL2`), the modeler must decide what the
outer count means, and the severity side `!` is the switch:

- `5000 claims 300 xs 0 sev agg.SL2 !` keeps the zero atom: 5000 **policies**,
  about 22 percent of which pay nothing. This equals the aggregate reinsurance
  benchmark (`SL2re` with `aggregate ceded to 300 xs 0`, referenced with no
  outer layers clause).
- `5000 claims 300 xs 0 sev agg.SL2` (default conditional) conditions the zero
  atom away: 5000 **loss bearing policies**. A different, legitimate model.

Docs positioning (author ruling 2026-08-16): **the reinsurance formulation is
not shown in the docs at all**; it misleads readers toward heavy machinery the
occurrence limit form replaces. It survives only as a test suite equivalence
pin. The severity side `!` is rare in practice, more pedagogy than production;
the docs present it as the switch that explains the semantics, not as a
recommended idiom.

**The zt identity** ties the representations together and belongs in both the
docs and the tests: conditioning the per policy aggregate on being positive is
the same as zero truncating the frequency, because for an almost surely
positive severity the events coincide. Concretely, with `agg SLzt ... poisson
zt` (no freq `!`, so 1.5 is the underlying Poisson parameter and the truncated
mean is about 1.93):

```
300 xs 0 sev agg.SL2      ==  300 xs 0 sev agg.SLzt      (same law, pinned by test)
```

while the freq `!` variant (`poisson zt !`, truncated mean held at 1.5) is a
genuinely different model. The docs table lays out the three representations
(zt inner; plain inner with severity `!`; plain inner conditional) and which
pairs coincide.

The guidance section also carries the two `!`s disambiguation of section 4.1
prominently: frequency `!` adjusts the zt/zm mean convention; severity `!`
keeps a layer unconditional; a reference program can carry both.

Single program flows work in one call (finding 10): a `build` or `build_many`
whose program defines `SL` and then uses `agg.SL` in a later statement
resolves correctly; no separate priming build is needed.

## 6. Author rulings recorded

2026-08-15:

1. `easy_update` was a plain routing alias for `update`; nothing special.
2. Renormalization of the materialized probabilities honors the inner update's
   effective `normalize` argument (the inner's choice via its hints, never the
   outer's).
3. No new build kwargs; `hints{}` is the sizing channel.
4. Commensurable grids are required: the inner bs should divide the outer bs,
   never incommensurable.

2026-08-16:

5. Conceptual model: a certified, fully formed dsev, referenced by name
   (section 2; terminology refined in ruling 15); the dsev representation
   query is nullary; dsev moments are reported as theoretical; the inner's
   theoretical tail is reported as the tail.
6. SeverityMeta: start afresh on the new foundation; do not reinstate the
   `meta.` keyword.
7. `as_severity` stays and must work on both ports and aggs (breakage
   reproduced live; repro recorded in section 4.9).
8. Hygiene safeguard proposed: referenced inners carry explicit `log2` and
   `bs` hints (confirmed firm in ruling 14).
9. Docs guidance: express inner and outer as simply as possible; prefer
   occurrence limits over reinsurance clauses; the split limit headline uses
   the zt inner with an outer occurrence limit.
10. A single `build_many` defining the inner and using it in a later statement
    must work.
11. Commensurability is a final query step at the end of bucket selection, not
    a fancy joint optimization.
12. Both `sev` and `ssev` accept references, with the compatibility matrix of
    section 4.2 (`sev` over a signed inner warns and clamps).
13. The two `!`s (frequency zt/zm mean convention versus unconditional layer)
    are distinct and the docs must keep them straight; the severity `!` is
    inert for inners without zero mass and load bearing for inners with it.

2026-08-16, third round:

14. The hygiene rule is firm (mandatory `log2` and `bs` hints on referenced
    inners), with `with_hints` on `Aggregate` and `Portfolio` as the
    certification helper (section 4.10).
15. Terminology: a named reference, parsed with the program and resolved
    subsequently; never textual injection (which is why only `agg.NAME` calls
    are allowed, and why a doubled `!` can never arise), and not object
    sharing (the inner may exist only as a recipe, for example in
    `library.agg`, never yet built).
16. The reinsurance formulation of the split limit is not shown in the docs
    (it misleads); it survives only as a test pin. The severity `!` is rare in
    practice, presented as pedagogy.
17. The tail descriptor stays reporting only: the inner's theoretical tail
    already drove the inner's own window choice, and the deficit warning is
    the numeric safeguard (section 4.6).

## 7. Phases

Each phase is its own version bump with a one line commit and a CHANGELOG
section, per house rules. The full gate (`uv run pytest -m 'slow or not
slow'`) runs at each bump; phases A, B, and B2 also run the numerics gate
(`-W error::RuntimeWarning`) since they touch discretization and sizing.

### Phase A `[SeverityMeta-Afresh]`

Section 4.9 plus the shared `_dhistogram_from_object` in `_severity.py` (built
here, consumed by phase B). Tests in the `tests/test_discrete_severity.py`
idiom: the author's port repro from section 4.9 passes; the same shape on a
bare agg via the new `Aggregate.as_severity`; `Severity(updated_agg)` builds
with exact moments matching the source's output pmf; a never updated source
raises; conflicting `sev_a`/`sev_b` raises; layering a meta severity uses
exact `layer_moments`; the signed clamp and warning of section 4.2 at the
programmatic level.

### Phase B `[Agg-As-Severity]`

The DecL feature: agg and port sources, `sev`/`ssev`/`!` forms, depth N, cycle
guard, hygiene check, tail descriptor. The commensurable sizing step follows
as phase B2. Touches: `decl.lark` (section 4.1), `parser.py` (handlers and
algebra guards, section 4.3), `underwriter.py` (`_sev_ref_stack`,
`_resolve_sev_ref`, `_factory` hook, sections 4.4 and 4.5), `decl_writer.py`
(section 4.3), `_aggregate.py` (tail reporting surfaces, section 4.6;
`with_hints`, section 4.10), `_portfolio.py` (`with_hints`).

Tests, new file `tests/test_agg_as_severity.py`, plain function idiom with
`from aggregate import build`, module docstring naming the label and version:

- exactness: an outer `dfreq [2] sev agg.Inner` over a discrete inner equals
  the self convolution of the inner density (`np.allclose`)
- **the split limit representations** (section 5): the plain inner with
  severity `!` under `300 xs 0` equals the aggregate reinsurance benchmark;
  the zt identity (`300 xs 0 sev agg.SL2` equals `300 xs 0 sev agg.SLzt`);
  and the freq `!` variant differs (three way pinning of the docs table)
- the headline program (zt inner, no severity `!`) builds and its severity has
  no zero atom
- hygiene: a reference to an inner without `log2`/`bs` hints raises, and the
  error text names `with_hints` and contains valid ready to paste hints
- `with_hints`: `build(inner.with_hints())` re registers the inner and a
  subsequent reference resolves with the pinned grid; extra keyword hints
  (for example `normalize=False`) appear in the emitted program; an object
  that has never been updated raises
- an outer carrying an `approximate` clause over a reference severity builds
  (smoke; the moment matched surrogate reads the dsev moments)
- single program: one `build` call defining `SL` and using it in `Auto.SL`
  works (finding 10)
- `sev port.NAME` matches a manual dhistogram built from the port total
- signed matrix: `ssev` over a signed inner keeps negative atoms; `sev` over a
  signed inner warns and clamps to the zero atom; `ssev` over a nonnegative
  inner is silent
- tail descriptor, pinning the section 4.6 derivation procedure: the
  `agg UB dfreq [1] sev gamma 100 cv 1` inner yields an outer reporting an
  unbounded right tail; the depth 2 chain (`OUTER2` over `OUTER` over `UB`)
  also reports unbounded (the transitivity pin: descriptor precedence over
  the finite dsev detachment); a finite outer layers clause over an unbounded
  inner reports bounded at the limit (the step 5 override); a `ceded to`
  aggregate cover inner reports bounded; a `net of` inner reports unbounded;
  a bounded severity under a Poisson frequency reports unbounded (the step 3
  second clause); a genuinely bounded inner (bounded severity, `dfreq`)
  reports bounded
- a depth 2 chain works (each level carrying its own hints); an A to B to A
  cycle and a redefinition self reference both raise with the chain named
- an unknown reference raises at parse time; `2 * agg.RefCarrier` raises the
  algebra guard
- iterated builds: building the outer twice works; redefining the inner
  between builds changes the outer (the re resolution policy pinned by test)
- `hints{normalize=False}` on the inner keeps the deficit
- a pnl engine smoke case whose engine aggregate uses a reference severity
- the unparser renders `sev agg.SL`, `ssev agg.SL`, and the `!` suffix
  (`format_program` round trip)
- `create_frequency()` on a reference severity object (verify the
  `_frequency_program` re parse is inert with `sev_ref` present)
- implementation check, settled during phase B: whether a `bvagg` body can
  parse a severity clause; if it can, a reference there either resolves
  through the same hook or raises a clear not supported error, with a test
  either way (the resolution hook currently covers agg, pnl, and xpnl kinds
  only)

Corpus and hygiene: three or four lines in `src/aggregate/agg/_test_suite.agg`
(definitions before uses, hints on the inners; the conftest preloads in file
order) with mirrors in `decl-testers.agg`; recapture
`tests/data/expected_specs.json` (only the added lines should move; any other
movement is a finding); grammar reference regen (`python -m aggregate.parser`);
narrative docs (section 9); `dev/TODO.md` updates (note under
`[Unparser-Reference-Gaps]` that the new reference kind renders as a reference
from day one; add the follow up in section 11); `dev/FEATURES.csv` regen via
`dev/regen_features.py`.

### Phase B2 `[Agg-As-Severity-Commensurable-Grids]`

The section 4.7 procedure, in its own bump immediately after phase B: the `d`
carrier attribute set in `_resolve_sev_ref`, the one snap function at the end
of `bs_window`, the `bs_window_df`/`bs_explanation` reporting, and the pinned
bs warning. Tests, added to `tests/test_agg_as_severity.py`:

- inner with `hints{bs=1/32}`; auto sized outer bs equals `(1/32) * 2**m` for
  some nonnegative integer `m`, and `bs_explanation` names the snap with the
  inner id and both bs values
- an outer whose moment estimate falls below `d` snaps to exactly `d`
- an integer lattice inner (`hints{bs=1}`, discrete atoms) under a `dfreq`
  outer: the exact discrete window still wins and no snap fires (the
  `d / b0` integer row of the decision table)
- an estimator result that is already an exact integer multiple of `d` is
  left alone (no forced power of two)
- pinned outer bs incommensurable with `d` warns, and the warning names the
  adjacent integer multiples of `d`; a pinned commensurable bs is silent
- the snapped case re derives origin and log2 through `_size` (asserted via
  the returned window quantities being self consistent: span covers the
  winning row's `x_hi`)

### Phase C `[Agg-As-Severity-Port-Units]`

Apply `_resolve_sev_ref` to portfolio unit specs carrying `sev_ref` in
`_factory`'s port branch, before `Portfolio(name, agg_list, uw=self)`. Tests:
a port whose unit uses `sev agg.X`; a corpus line; snapshot recapture.

## 8. What this does not change

- No new terminals and no new keywords (and no `meta.` revival): no
  `_TERMINAL_LABELS` additions, no Pygments or sublime syntax edits, and
  nothing owed to the app side `decl-keywords.json` mirror.
- No change to the `sev.NAME` inlining path or to any existing spec's
  snapshot.
- `Portfolio.as_severity` keeps its public surface (its numerics improve in
  phase A); `Aggregate.as_severity` is the one new public method, mirroring
  it.

## 9. Documentation

- Grammar reference regen as part of the phase B bump.
- A narrative subsection in the DecL severity documentation: the reference
  form, the reference model of section 2 (including the mandatory hints rule,
  `with_hints`, and the error UX), the P(0) and `!` semantics with the two
  `!`s disambiguation, the zt identity, the signed matrix, the
  commensurability behavior, the tail reporting exception, and the accuracy
  story of section 4.8.
- The user guidance of section 5 (simplicity preference order, occurrence
  limits before reinsurance, linking the existing reinsurance versus exposure
  occurrence limits discussion) with the split limit worked example and the
  three representation table in `docs/2_aggregate_overview/features.rst`. The
  reinsurance formulation of the split limit does not appear in the docs; the
  severity `!` appears as pedagogy, not as a recommended idiom.
- `as_severity` and `with_hints` documented for both classes.
- Docs rebuild is left pending per house rule; `.rst` edits land in lockstep
  with the code.

## 10. Risks and watch items

- The mandatory hints rule is the main new friction point; `with_hints` and
  the ready to paste error message are the mitigations.
- `BivariateAggregate` sits outside the resolution hook; the phase B
  implementation check (its test list) ensures a reference in a bvagg body
  cannot silently miss resolution.
- The severity `!` versus frequency `!` confusion is the likeliest user trap;
  the docs table and the three way representation test are the mitigations.
- Definition before use ordering is required for references, identical to
  every existing dotted reference; a `to_agg` export that reorders entries can
  break re import. The same class of issue exists today; document, do not fix
  here.
- Direct `Aggregate(**recipe_spec)` bypassing the underwriter gives
  `TypeError` on `sev_ref`. Terse but truthful; documented.
- Atom counts up to `2**log2` minus dropped zeros flow into
  `validate_discrete_distribution`, `_DiscreteRV`, and the rebucket; all
  vectorized, measured fine at 64k, and the zero drop usually shrinks the
  count drastically.
- A new grammar ambiguity would surface in `test_grammar_ambiguity` (the exact
  `KNOWN_AMBIGUOUS` sweep is the arbiter); predicted clean because the new
  alternatives are token disjoint.
- The tail descriptor (section 4.6) touches reporting surfaces not designed
  for a severity with two tail stories (discrete actual, theoretical
  reported); keep the descriptor read only metadata and resist letting it leak
  into numerics.
- Reworking `SeverityMeta` changes `Portfolio.as_severity` numerics (hybrid
  histogram to exact atoms). No tests cover the old behavior, and it is
  currently broken in practice anyway (the author's repro); the CHANGELOG
  entry states the change plainly.

## 11. Follow ups recorded in dev/TODO.md at phase B

- `[Agg-As-Severity-Result-Cache]`: cache built inner objects across iterated
  outer builds (and within a single program that both defines and uses an
  inner), only if profiling shows the fresh re resolution cost matters.

## 12. Name vetting (rg verified against the tree; re verify at implementation)

| Name | Kind | Result |
|---|---|---|
| `sev_ref` | spec key | clean; `dev/TODO.md` proposes exactly this name (alignment, not collision) |
| `sev_clause_ref`, `sev_clause_ref_signed`, `sev_ref_agg`, `sev_ref_port`, `sev_ref_uncond` | grammar aliases and transformer methods | clean; echo the existing `agg_source_ref_agg` and `agg_source_ref_port` idiom; re vet the final factoring |
| `_resolve_sev_ref` | Underwriter method | clean; joins the `_resolve_hints` family; no attribute or method collision on `Underwriter` |
| `_dhistogram_from_object` | module function, `_severity.py` | clean, zero matches |
| `_sev_ref_stack` | Underwriter attribute | clean, zero matches |
| `as_severity` | new `Aggregate` method | exists only on `Portfolio` today; no `Aggregate` attribute or method collision found; final vet at implementation |
| `with_hints` | new method on `Aggregate` and `Portfolio` | no match in the tree; distinct from the `hints` spec key, the `hints` constructor parameter, and `_parse_hints`/`_resolve_hints` (a method name does not shadow the attribute `hints`); final vet at implementation |
| severity reference cycle message | error text | matches no existing message; chain rendered `agg.A -> agg.B -> agg.A` |

Names still to be chosen and vetted before coding, called out here per the
house rule: the tail descriptor attribute (section 4.6), the frequency
boundedness helper (section 4.6 step 2), the `d` carrier attribute on the
materialized severity and the snap function in `_bucket_window.py` (section
4.7), and the warning categories for the signed clamp, the deficit, and the
incommensurable pin (sections 4.2, 4.5, and 4.7).

---

## 13. Execution notes (written as the phases landed)

Read this before working on anything the plan touches: it records where the
implementation diverged from the design above, and what executing it turned up
that the design did not anticipate. Section numbers refer to the plan.

### Phase A `[SeverityMeta-Afresh]`, 1.0.0a290

1. **`SeverityMeta` became a subclass of `SeverityDHistogram`.** Section 4.9
   item 3 says "routes through the `SeverityDHistogram` machinery"; subclassing
   is the cleanest form of that. `_build` fills `sev_xs` / `sev_ps` from
   `_dhistogram_from_object` and delegates, exactly as `SeverityFixed` does, so
   the exact moments, `support_atoms`, `_DiscreteRV` and the negative atom auto
   sign all come for free and cannot drift from the plain `dsev` path.
2. **The author's `IndexError` was diagnosed at `_severity.py:1025`**, not in
   the known broken branch. `Severity.__init__` passed
   `name=sev_name if isinstance(sev_name, str) else ''` to
   `ss.rv_continuous.__init__`, and scipy indexes `name[0]` to pick the article
   for its generated docstring. Every object valued `sev_name` (the `meta` and
   `copy` paths both) hit it before any severity logic ran. The empty string is
   now the class name.
3. **`long_name` was made readable for an object valued `sev_name`.** Not in
   the plan; `info` printed a whole `Portfolio` report into its one line
   "severity distribution" row.
4. **`sev_a` / `sev_b` are accepted when they restate the source's current
   grid** and raise only when they contradict it. Section 4.9 item 2 says
   "retired"; a hard raise would have broken `Portfolio.as_severity`'s own call
   shape for no gain, and that call now simply does not pass them.
5. **Deficit floor**: `REFERENCE_DEFICIT_MATERIALITY = 1e-6` per section 4.5,
   named as a module constant in `_severity.py` so the choice is visible. It is
   deliberately 100 times tighter than the house `DEFICIT_MATERIALITY`, because
   a severity deficit is multiplied by the outer frequency.
6. **Warning categories** (section 12's open list): plain `UserWarning` for the
   signed clamp, `DefectiveDistributionWarning` for the deficit (it *is* a
   defective distribution, so the house category fits), plain `UserWarning` for
   the conditioning warning of note 12 below. No new warning class was added;
   none of the three earns one.

### Phase B `[Agg-As-Severity]`, 1.0.0a291

7. **Names chosen** (section 12's open list). On `Severity`, class level
   defaults so every reader can test them: `reference_id` (the dotted id, `''`),
   `reference_support_max` (the theoretical upper end, `None`), `reference_bs`
   (the source's own bs, `None`). In `tail.py`, `output_support_max` plus the
   private `_cession_cap` / `_retention_cap` / `_cover_max`. On `Underwriter`,
   `_resolve_sev_ref`, `_sev_ref_hygiene_message`, `_warn_sev_ref_conditioned`,
   `_sev_ref_stack`; module level, `_carries_sev_ref` and `_stamp_sev_ref`.
8. **The tail descriptor needed a switch, not an attribute read.** Section 4.6
   says reporting only, and wiring `reference_support_max` into
   `tail.severity_support` unconditionally would NOT have been: the sizer reads
   the same rows through `Aggregate._loss_tail_classes`, and an unbounded
   descriptor turns the aggregate's right tail thick, which fires the single big
   jump floor and changes the grid. So `severity_support`, `severity_tail_row`,
   `combined_severity_row` and `build_tail_rows` take a `reference=` flag,
   `Aggregate._tail_rows` defaults it to `True`, and `_loss_tail_classes` passes
   `False`. That is the plan's intent implemented exactly: reporting reads the
   reference's law, numerics ride the atoms.
9. **`Aggregate.bounded` and `Severity.bounded` were left alone**, per section
   4.6's "consumers pinned for v1: `tail_behavior_df` only". They derive from
   `tail._severity_bounded`, which the sizer also reads through
   `Aggregate._bounded_severity_window`, so widening them would have leaked into
   numerics by the same route as note 8. **The consequence is a visible
   disagreement**: for `agg X dfreq [2] sev agg.Unbounded`, `tail_behavior_df`
   reports an infinite max while `X.bounded` is `True`. The aggregate row's note
   names the source and says it was sized on its atoms, so the two readings are
   legible side by side, but the author may want `bounded` widened later. Doing
   it properly means threading `reference=` through `classify_severity` and
   `aggregate_tail_info` as well, and deciding what `_bounded_severity_window`
   should then believe.
10. **Frequency boundedness (section 4.6 step 2) reuses
    `Aggregate._frequency_count_support`** rather than a new lookup helper. One
    divergence follows: that method returns `(0, inf)` for a **binomial** count,
    where the plan calls binomial bounded. The reading is conservative
    (unbounded), it is what the source's own `tail_behavior_df` already says, and
    keeping the two consistent matters more than the extra precision. Fixing it
    means widening `_frequency_count_support`, which moves existing tail output.
11. **A limit profile over a reference takes `max(exp_limit)`.** The descriptor
    survives only under an infinite outer limit (step 5), and a vector
    `exp_limit` is reduced with `np.max`, so a profile mixing a finite and an
    infinite limit reports unbounded for every component. Not a case the plan
    considers; the alternative is per component stamping, which the materialized
    `SeverityDHistogram` cannot support because `_build` overwrites `limit` with
    the largest atom.

### What executing it turned up (none of this is in the plan)

12. **The headline program of sections 1 and 5 needed a severity side `!`.**
    The plan reasons that a zero truncated inner has no mass at zero so no `!`
    is needed. True of the theoretical law, false of the materialized `dsev`:
    any severity with positive density at the origin discretizes mass into the
    first bucket, and `gamma 50 cv 2` is shape 0.25, so about 10% of one claim
    lands below `bs / 2` and the per policy aggregate materializes with about 7%
    at its zero atom. The default conditional layers clause rescales that away,
    lifting the severity mean from 44.31 to 47.52 and the outer answer by 6%.
    **The behavior is unchanged** (it is exactly what a hand written `dsev` with
    a zero atom gets), the docs now write the headline with `!`, and the
    resolver warns where the reading cannot have been intended, namely when the
    source's claim count is never zero. A plain Poisson inner really does have
    `P(S = 0) > 0` and section 5's conditioning idiom is a legitimate model, so
    that stays silent. **Open for the author**: whether a reference severity
    should default to unconditional under a layers clause. That is a language
    semantics change, so it was not taken here; tracked as
    `[Reference-Severity-Zero-Atom-Default]` in `dev/TODO.md`.
13. **`hints{}` leaked between the statements of one program**, a pre existing
    `build_many` bug sitting directly in this feature's path. The update loop
    rebound `log2` / `bs` / `bucket_sizing_p` / `kwargs` in place, so the hints
    of statement 1 became the caller defaults for statement 2: a first statement
    carrying `hints{log2=16; bs=1/32}` built every later statement on that grid.
    Since the hygiene rule guarantees a referenced inner carries hints, and
    finding 10 of section 3 promises the single program define then use flow
    works, this had to be fixed for the plan's own test to mean anything. The
    loop now holds the caller's arguments fixed and copies `kwargs` per output.
14. **`interpret_file` could not validate a file that declares a name and then
    uses it.** It parsed statement by statement without registering anything, so
    the new corpus lines made it raise `KeyError` out of `_safe_lookup` and abort
    the whole run, which is the opposite of what a per statement error collector
    should do. It now parses against a scratch underwriter seeded from the
    caller's recipes and registers each statement as it goes; `LookupError`
    joins the caught set so an unresolvable reference is an error row. It also
    no longer mutates the calling underwriter, which the old hand seeded
    `sev One` did.
15. **`test_decl_unparser` parsed against `_test_suite.agg` alone** while its
    corpus is three files, so a reference defined in `decl-testers.agg` could not
    resolve from `decl-testers.agg`. It now preloads the whole corpus, lazily
    (the parse costs about 11 seconds and an import time preload pays it in
    every xdist worker) and tolerantly (section X of `decl-testers.agg` is
    intentional parse errors, so `databases=` cannot be used). Verified
    beforehand that the three files have no `(kind, name)` collisions, so
    nothing shadows anything.
16. **The two split limit representations agree on the severity's moments to
    1e-13 and on the aggregate mean to about 4e-4.** The residual is the
    discretization path, not the semantics: an unlimited reference takes the
    mean preserving linear scatter and reproduces the severity mean exactly,
    while an outer layers clause puts it on the survival difference path, which
    carries a positional bias that shrinks with the grid. This is ordinary
    `dsev` behavior and it is the numeric argument behind section 5's preference
    for the occurrence limit form. The test pins the moments exactly and the
    aggregate to 1e-3.
17. **Two pre existing failures, unrelated to this work**, found by the gates
    and confirmed at `c53db28` by stashing: `test_massive_pnl_one_sweep_ledger`
    fails with `TypeError: Index must be a MultiIndex` (a `slow` test, so the
    everyday suite never runs it), and
    `docs/2_aggregate_overview/features.rst` has one `ipython` block that does
    not execute (`book_pnl.economic_df`). Neither is touched here.
