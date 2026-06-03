# Split `note{...}` into pure notes + a new `hints{...}` settings clause

**Status:** planning only. No `src/` changes made.
**Date:** 2026-06-03

## 1. Motivation

`note{...}` is currently overloaded: besides free-text annotation it doubles as
a `key=value;` side-channel for build settings, parsed by
`underwriter._parse_note`. Any `=` in note prose (e.g. `x_min<=-6`) is
mis-read as a kwarg and crashes the build:

```
agg PnL.Fix dfreq [3] dsev [-2 5] [.5 .5] note{... needs x_min<=-6}
→ TypeError: update_work() got an unexpected keyword argument 'needs x_min<'
```

**Decision:** make `note{...}` a *pure* annotation and introduce a dedicated
`hints{...}` clause that carries *only* `key=value;` build settings.

### Settled design decisions (from review chat)
- **Name:** `hints`.
- **Back-compat:** clean break — notes become pure immediately; emit a
  **deprecation warning** if a note still looks like it carries settings; migrate
  the corpus now.
- **Unknown keys:** **warn + ignore** (validate against an allow-list; never
  crash on an unrecognized key).
- **Precedence:** **caller always wins** — explicit `build(...)` kwargs override
  in-program `hints`, uniformly (including `recommend_p`, fixing today's quirk
  where a note's `recommend_p` overrode the caller).
- **Separator:** `=` (matches existing convention; avoids `:` clashing with
  slice ranges like `[1:6]`).
- **Ordering:** `note` and `hints` are both optional and order-free relative to
  each other; at most one of each. Duplicate key within a clause, or a duplicated
  clause → **warn, last wins**.
- **Scope:** allowed everywhere `note` is today (agg, sev, port).
- **Value parsing:** generic inference — `int` / `float` / `a/b` fraction
  (keeps `bs=1/64`) / `True`/`False` / else string.

## 2. Grammar changes (`decl.lark`)

### 2a. New terminal (next to `NOTE.3`, ~line 289)
```lark
HINTS.3: /hints\{[^}]*\}/
```
Same shape/priority as `NOTE`. Brace-delimited, so the whole clause is one
token and its internals are parsed by us (not Lark) — no interaction with the
Earley/dynamic lexer for the `=`/`;`/`/` inside.

### 2b. New rules (replace the `note` rule block, ~lines 158-162)
```lark
// Optional trailing annotation + settings, order-free.
hints: HINTS  -> hints_some
     |        -> hints_none

// `trailer` groups the two optional clauses in either order. Unambiguous
// because NOTE and HINTS have disjoint token prefixes (`note{` vs `hints{`).
trailer: note hints   -> trailer_nh
       | hints note    -> trailer_hn
```
Keep the existing `note: NOTE -> note_some | -> note_none`.

> `note hints` with both optional already covers "note only", "hints only",
> "neither", and "both in note-first order"; the second alt adds hints-first.
> The four parse combinations are distinct, so no ambiguity.

### 2c. Thread `trailer` into the output rules
Replace the trailing `note` in each `agg_out` / `sev_out` variant with
`trailer`; for `port_out` replace its mid-rule `note` (it sits *after* `name`,
before `agg_list` — keep that position):

```lark
port_out: PORT name trailer agg_list

agg_out: AGG name exposures layers sev_clause occ_reins freq agg_reins trailer -> agg_out_full
       | AGG name dfreq      layers sev_clause occ_reins         agg_reins trailer -> agg_out_dfreq
       | AGG name TWEEDIE expr expr expr trailer                                   -> agg_out_tweedie
       | AGG name builtin_agg occ_reins agg_reins trailer                          -> agg_out_rename
       | builtin_agg agg_reins trailer                                            -> agg_out_builtin

sev_out: SEV name sev trailer    -> sev_out_sev
       | SEV name dsev trailer   -> sev_out_dsev
```

## 3. Parser/transformer changes (`parser.py`)

### 3a. Terminals
- Keep `NOTE` (strips `note{` / `}` → inner text), ~line 271.
- Add `HINTS(self, tok)` → `str(tok)[6:-1]` (strip `hints{` / `}`).

### 3b. New transformer methods
```python
def hints_some(self, c): return c[0]      # raw inner text
def hints_none(self, c): return ""

def trailer_nh(self, c):  # (note, hints)
    return {"note": c[0], "hints": c[1]}
def trailer_hn(self, c):  # (hints, note)
    return {"note": c[1], "hints": c[0]}
```
`trailer` returns a small dict so the order-free variants collapse to one shape.

### 3c. Update the 8 output methods that unpack `note`
Each currently ends `..., note = c` and sets `spec["note"] = note`. Change to
unpack the `trailer` dict and store both:
```python
*_, trailer = c
spec["note"]  = trailer["note"]
spec["hints"] = trailer["hints"]     # raw string; parsed in underwriter
```
Affected: `port_out`, `agg_out_full`, `agg_out_dfreq`, `agg_out_tweedie`,
`agg_out_rename`, `agg_out_builtin`, `sev_out_sev`, `sev_out_dsev`.
(`agg_out_tweedie` already builds its own note string — append `hints` likewise.)

The spec dict gains a `"hints"` key everywhere `"note"` exists; default `""`.

## 4. Settings parsing (`underwriter.py`)

### 4a. Replace `_parse_note` with `_parse_hints`
New function parses the **hints** string (not the note) into a typed dict:

```python
_HINT_KEYS = {                      # allow-list = build/update knobs
    'log2', 'bs', 'padding', 'normalize', 'recommend_p',
    'sev_calc', 'discretization_calc', 'force_severity', 'x_min', 'x_max',
}

def _parse_hints(txt):
    """Parse a `hints{...}` body 'k=v; k=v' into a typed kwargs dict.

    Generic value inference: int, float, `a/b` fraction, True/False, else str.
    Unknown keys -> warning, dropped. Duplicate key -> warning, last wins.
    """
    kw = {}
    for chunk in txt.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.count('=') != 1:
            logger.warning("hints: ignoring malformed clause %r", chunk)
            continue
        k, v = (s.strip() for s in chunk.split('='))
        if k not in _HINT_KEYS:
            logger.warning("hints: unknown key %r ignored "
                           "(known: %s)", k, sorted(_HINT_KEYS))
            continue
        if k in kw:
            logger.warning("hints: duplicate key %r; last value wins", k)
        kw[k] = _coerce_hint_value(v)
    return kw
```
`_coerce_hint_value`: try `int`, then `float`, then `a/b` → `float(a)/float(b)`,
then `True`/`False`, else leave as `str`. (Replaces the per-key coercion and the
always-true `elif 'normalize'` bug in the old code.)

### 4b. Merge with **caller wins**
In `build_many` (the two `_parse_note` call sites, ~lines 624 & 660) and
`build`, after parsing:
```python
hints = _parse_hints(d.get('hints', ''))
# caller wins: only fill keys the caller did NOT set.
if 'log2' in hints and log2 == 0:           log2 = int(hints.pop('log2'))
if 'bs'   in hints and bs   == 0:            bs   = hints.pop('bs')
if 'recommend_p' in hints and recommend_p == RECOMMEND_P:
                                             recommend_p = hints.pop('recommend_p')
for k, v in hints.items():                   # remaining update kwargs
    kwargs.setdefault(k, v)                  # setdefault == caller wins
```
`setdefault` is the whole "caller always wins" rule for the pass-through keys;
the sentinel checks (`log2==0`, `bs==0`, `recommend_p==RECOMMEND_P`) implement it
for the three special ones.

### 4c. Deprecation warning for settings-in-notes
After splitting note/hints, scan the **note** for the old pattern and warn (do
**not** act on it):
```python
if re.search(r'\b(log2|bs|padding|normalize|recommend_p)\s*=', d.get('note', '')):
    logger.warning(
        "note{...} no longer sets build options; move 'key=value' settings "
        "into a hints{...} clause. The note is now treated as pure text.")
```

## 5. Corpus migration (8 lines)

Move settings out of notes into `hints{}` in `src/aggregate/agg/test_suite.agg`
and `test_decl.agg`. Known offenders (from grep): `normalize=False`,
`log2=17` (×2), `bs=1`, `log2=8`, `padding=1`, `log2=18`, `bs=500`. Mechanical:
`note{... ; log2=18}` → `note{...} hints{log2=18}` (and keep `test_decl.agg` in
sync per project rule). `expected_specs.json` snapshot will gain `"hints"` keys
— regenerate or hand-patch the affected entries (see §6).

## 6. Tests & blast radius

- **Spec-shape snapshot.** Every corpus line now parses to a spec dict with an
  extra `"hints"` key (default `""`). `tests/data/expected_specs.json` must be
  re-captured (or patched to add `"hints": ""` / the migrated values). This is
  the main mechanical churn — expected and benign.
- **New parse tests** (append DecL to `test_decl.agg`, per project rule):
  - `note{}` only; `hints{}` only; both orders; neither.
  - `hints{log2=18; bs=1/64}` → typed dict `{log2:18, bs:0.015625}`.
  - prose with `=`/`<=`/`;` in `note{}` builds fine (the original bug repro:
    `dsev [-2 5] [.5 .5] note{... needs x_min<=-6}`).
  - unknown key `hints{frobnicate=3}` → warns, ignored, still builds.
  - duplicate `hints{bs=1; bs=2}` → warns, `bs==2`.
  - caller precedence: `build(prog_with_hints_log2_18, log2=20)` → `log2==20`.
- **Functional regression.** The `tests/baseline/` parquet densities are keyed
  on built objects, not spec text; with corpus migrated to equivalent `hints`,
  the built grids must be **bit-identical** (the same log2/bs reach `update`).
  Confirm empty diffs.
- **Round-trip / `__str__`.** If any code reconstructs a program string from a
  spec (program rendering / `repr`), teach it to emit `hints{}` and append a
  reconstruction test.

## 7. Open implementation notes / risks

- **`port_out` note position.** Port's note is mid-rule (`PORT name note
  agg_list`); the `trailer` keeps that slot (`PORT name trailer agg_list`), so a
  port's note/hints sit *between the name and the agg list*, order-free with each
  other — not at the very end. Document this asymmetry in the grammar comment and
  the language docs.
- **Earley ambiguity check.** `trailer: note hints | hints note` is unambiguous
  given disjoint `note{`/`hints{` prefixes, but run the full parse suite to be
  sure the dynamic lexer doesn't introduce a surprise with the empty-production
  alternatives. If any ambiguity warning appears, collapse to a single
  `trailer: clause clause?` over a `clause: note | hints` union and dedupe in the
  transformer.
- **`build` vs `build_many`.** Wire `_parse_hints` into the same spots that call
  `_parse_note` today (both `build_many` branches, ~624/660). Confirm the
  `tweedie` and `builtin`/`rename` paths carry `hints` through too.
- **Docs.** Update the DecL language reference (`.rst`) and the `note{}` section
  to describe `hints{}`, the allow-list, precedence, and the deprecation. Keep
  `.rst` edits in lockstep; do not trigger a full docs build in the loop.

## 8. Summary of file touch-list
- `src/aggregate/decl.lark` — `HINTS` terminal, `hints`/`trailer` rules, thread
  `trailer` into 8 output rules + `port_out`.
- `src/aggregate/parser.py` — `HINTS` terminal handler, `hints_*`/`trailer_*`
  transformers, update 8 output methods to emit `spec["hints"]`.
- `src/aggregate/underwriter.py` — replace `_parse_note` with `_parse_hints` +
  `_coerce_hint_value`; caller-wins merge; note deprecation warning.
- `src/aggregate/agg/{test_suite,test_decl}.agg` — migrate 8 settings-in-notes.
- `tests/data/expected_specs.json` — re-capture (adds `"hints"` key).
- `tests/test_*` / `test_decl.agg` — new parse + precedence + warning cases.
- docs `.rst` — language reference for `hints{}` (rebuild deferred to author).

The original failing program builds for free the moment notes go pure: the
`<=` in its prose is no longer parsed.
