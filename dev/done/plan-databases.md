# plan-databases — make Underwriter database loading legible

> **STATUS: DONE — shipped in 1.0.0a32 (2026-06-06).** All four phases landed
> together: dict-backed store + `source` provenance, the glob-aware resolver,
> `load` / `reload` / `resolve_databases` / `available_databases`, the
> `_loaded` flag, `databases` = loaded paths, one preprocessing owner
> (`UnderwritingLexer.preprocess`), and `to_agg` save/export. `read_database` /
> `read_databases` renamed outright to `load` (no aliases). Full suite green
> (1018 passed); `src` ruff clean. Maps to TODO **H6**. Tagged `[A]` (alpha) —
> small, self-contained hygiene with zero numeric blast radius (same knowledge,
> clearer plumbing). The design notes below are kept as the as-built record.

## How the new design works (at a glance)

The constructor argument `databases=` says **what to load** (the request); the
attribute `uw.databases` reports **what actually loaded** — a list of resolved
file `Path`s, so you can always see *where* each one lived.

**Construction**

- `Underwriter(databases=None)` — load nothing. `uw.knowledge` is empty;
  `uw.databases == []`.
- `Underwriter()` — load the configured request (`config.build.databases`,
  default `('test_suite',)`): the bundled `test_suite.agg`.
- `Underwriter(databases='all')` — every `.agg` in the user dir
  (`~/.aggregate`) **and** the bundled dir. (`'default'` = bundled only;
  `'user'` = user dir only.)
- `Underwriter(databases='myfile.agg')` — that one file, found via the search
  path **cwd → user → default**. A glob (`'cat_*'`) loads every match across all
  three dirs (union).

Loading is **lazy**: nothing is read until you touch `uw.knowledge` (or call
`uw.load()`), so `import aggregate` stays cheap.

**Discover → load → inspect → reload**

- **Discover what's on disk:** `uw.available_databases()` — the `.agg` files
  found in cwd, `~/.aggregate`, and the bundled dir (name + where each lives).
- **Preview before loading:** `uw.resolve_databases('cat_*')` — the list of
  `Path`s that request *would* read, without loading. Answers "if I ask to load
  `xxx`, what will I get?"
- **Load one or more now:** `uw.load('mybook.agg')` or
  `uw.load(['cat_*', 'extra.agg'])` — resolves, reads, adds to the knowledge
  base, and appends the files to `uw.databases`.
- **See what loaded:** `uw.databases` (resolved file paths) and `uw.knowledge`
  (parsed entries, indexed by `(kind, name)`).
- **Create new:** `build('agg MyNew …')` adds an entry to the union, tagged
  `source='session'` (it is not in any file yet).
- **Save what you built:** `uw.to_agg('mybook')` writes the session entries to
  `~/.aggregate/mybook.agg` (user dir unless the path is absolute), re-loadable
  by name later. `to_agg(path, pattern, kind, source)` selects by name regex,
  kind, and provenance (default: everything built this session).
- **Reset to as-created:** `uw.reload()` — clears the knowledge base and
  re-reads the original request (drops in-session builds and ad-hoc loads).

**Error policy:** a file you name *explicitly* in `load(...)` that does not
exist **raises**; a glob or the configured set that matches nothing **warns**
(load-what's-there).

That is the whole intended surface: construct (with a request), `load`,
`resolve_databases` (preview), `available_databases` (discover), `databases` /
`knowledge` (inspect), `to_agg` (save), `reload`.

## The problem

"Reading databases" is spread across one overloaded attribute and two
similarly-named methods, with a fragile lazy-load trigger and a heavyweight
store. Coming back cold, it is hard to answer simple questions: *what will load?
when? from where? did it actually load?*

## Current behavior (the map)

- **`self.databases`** — one untyped attribute holding the *request*. It can be:
  the `_UNSET` sentinel (→ config `build.databases`), `None` (→ `[]`, load
  nothing), a single string, or an iterable. Within that, a string can be a
  **named collection** (`'default'` = bundled `.agg`, `'user'` = `~/.aggregate`
  `.agg`, `'all'` = both), a removed token (`'site'` → raises), or an
  **individual file** path. Collections and file paths share one untyped list.
- **`read_databases()`** — re-normalises `self.databases` (its own `== 'all'` /
  `isinstance str` logic, *separate* from the constructor's) into a file list by
  globbing `default_dir` / `user_dir` or appending bare entries, then calls
  `read_database` per file.
- **`read_database(fn)`** — singular: adds `.agg` if no suffix, searches
  cwd → `user_dir` → `default_dir`, reads text, **munges whitespace with its own
  regexes** (`^  +`→tab, `' +'`→`' '`), then `_interpret_program`. Not-found
  logs an error and silently returns.
- **`knowledge` property** — lazy: reads the configured databases **iff**
  `len(self._knowledge)==0 and self.databases`. Returns the
  `(kind,name)`-indexed DataFrame sorted, columns `['program','spec']`.
- **Store** — `self._knowledge`, a `pd.DataFrame` indexed by `(kind, name)`,
  columns `spec`, `program`.

## Critique (best-practice lens)

1. **`databases` is overloaded.** It conflates three different things — a load
   *intent* (`default`/`user`/`all`), individual *file paths*, and the "load
   nothing" states (`None`, `[]`) — in one untyped value, and the normalisation
   lives in **two** places (`__init__` vs `read_databases`) that can drift.
2. **Fragile lazy trigger.** `len(_knowledge)==0` as the "not loaded yet" signal
   is a proxy, not a fact. Consequences: a source legitimately yielding 0
   entries would re-trigger; **setting `build.databases = [...]` after first
   access never loads** (the frame is non-empty, so the guard is False); and
   `__repr__` has to deliberately dodge `self.knowledge` to avoid a hidden read
   — a tell that a property has surprising side effects.
3. **Singular/plural naming.** `read_database` vs `read_databases` differ by one
   letter for two quite different jobs (read one file vs resolve-and-read the
   configured set).
4. **Two preprocessing layers.** `read_database`'s regex munging overlaps
   `UnderwritingLexer.preprocess` (which already folds `\n\t` / `\n    `
   continuations and strips comments). The `' +'`→`' '` collapse is a second,
   global whitespace pass with no clear owner (and it rewrites inside `[ ]`
   vectors too).
5. **Inconsistent error policy.** `'site'` *raises*; a missing file *logs and
   continues*; an `OSError` *logs and continues*. A typo'd database name thus
   silently loads nothing.
6. **Heavyweight store.** A DataFrame as a `(kind,name)`→spec map forces
   `.xs(...)`, `.loc[[item]]`, `reset_index().iloc[0]`, a `# TODO fix unhashable
   slice`, and a `sort_index()` on every `.knowledge` access. A plain dict is
   simpler, faster, and clearer; the DataFrame is only really needed as a
   *display/`discover`* view, which can be built on demand.
7. **No record of what loaded.** After resolution the contributing files are
   forgotten, so "where did this entry come from / what did I load" is
   unanswerable.
8. **Mixed responsibilities.** Source resolution + file I/O + text munging +
   the knowledge store + parsing bridge + object factory all live on one class;
   database loading is a separable concern.

## Proposed design

A single, explicit loading pipeline over a dict-backed store, with one
resolver, one preprocessing owner, and an honest "loaded" flag.

### D1 — store: dict-backed, DataFrame on demand
Back the knowledge base with `self._knowledge: dict[tuple[str,str], ParsedProgram]`
(key `(kind, name)`). `__getitem__` / `_safe_lookup` become plain dict lookups
(kills the `.xs`/`unhashable slice`/`iloc[0]` code). Add a `source` field to
`ParsedProgram` (the originating file `Path`, or the sentinel `'session'` for
in-session builds) — provenance, see *The conceptual model*. `knowledge` returns
a DataFrame **built on demand** from the dict for display/`discover` — same
shape as today (`(kind,name)` index, `program`/`spec` columns; `source`
available as an optional column) so `discover` is unchanged. *Blast radius:*
`tests/conftest.py` and
`tests/capture_sly_snapshot.py` currently **write** via
`uw.knowledge.loc[(kind,name),:] = [...]`; they'd move to a small public
mutator — see D6.

### D2 — one typed, glob-aware resolver
`_resolve_database_request(req) -> list[Path]` — the *single* normaliser, used
by both construction and (re)load. The key idea: the **named collections become
predefined globs**, so there is one mechanism, not a special case per token.

- `None` / `[]` → nothing.
- **Reserved collection names** (sugar): `'default'` → `<default_dir>/*.agg`,
  `'user'` → `<user_dir>/*.agg`, `'all'` → both.
- **Everything else is a path or glob**, resolved against the search dirs
  **cwd → user_dir → default_dir** (all three globbed, incl. cwd; local-wins: a
  local copy overrides a bundled one with the same `(kind,name)`):
  - **Glob** (entry contains `*`, `?`, or `[`): glob all three search dirs and
    **union the matches** *(confirmed: union, and cwd is included)*, restricted
    to `*.agg` (a bare pattern with no suffix globs `<pat>.agg`, e.g. `cat_*` →
    `cat_*.agg`). An empty match **warns** — a glob means "load whatever is
    there."
  - **Literal** (no glob metacharacters, e.g. `test_suite`): add `.agg` if no
    suffix; **first match wins** across the search dirs (the nearest file). A
    missing literal in the *configured* request warns; the same name passed to an
    explicit `load(...)` call **raises** (D7 — the caller named one file).
  - **Absolute path / anything with a directory separator**: used as given (glob
    its basename if it contains metacharacters); the search dirs are *not*
    consulted.

The reserved `'site'` token (a removed "office-wide" third layer between user and
default) is **deleted** — not deprecated — since the functionality is gone.

This keeps the config default `('test_suite',)` resolving exactly as today, and
gives `databases=['cat_*']` (load all catastrophe books) or
`load('~/work/mybook.agg')` for free.

### D3 — honest lazy load
Replace the `len(_knowledge)==0` proxy with an explicit `self._loaded: bool`.
The bare `load()` (no argument) is the idempotent lazy path: resolve the stored
request, read each file once, set `_loaded=True`; `knowledge` calls it on first
access. Keep lazy (justified: the import-time module `build` must not parse
`test_suite.agg` eagerly), but make it a *fact*, not a heuristic — so `__repr__`
can read state without triggering I/O.

### D4 — the public surface (renamed outright)
State split (resolves the old overloaded `databases`):

- `self._request` — *private*: the load specification (the constructor
  `databases=` argument, or the config default). What `reload`/lazy-load resolve.
- `self.databases` — *public*: `list[Path]` of files **actually loaded**, with
  full paths. Populated as files are read; this is the "what / where loaded"
  view the author wants. *(Confirmed: `databases` reports loaded files, not the
  request; no separate `loaded_files` attribute.)*

Methods (renamed outright — no deprecated `read_database(s)` shims; `build` is
effectively the only caller and we are pre-1.0 alpha):

- `load(request=None)` — the one load verb. `request=None` loads the configured
  `self._request` (the lazy path); a given `request` (filename / glob /
  collection / list) resolves and reads *additively*, appending to
  `self.databases`. Reading "one or more" is just a glob or a list.
- `reload()` — **reset to as-created**: clear the store, `self.databases`, and
  `_loaded`; restore the original `self._request`; re-resolve and re-read. Drops
  ad-hoc `load(...)`-ed files and in-session `build(...)` entries. This is what
  `_refresh_default_underwriter` and config reload want.
- `resolve_databases(request=None)` — **dry run**: return the `list[Path]` a
  request *would* load, without reading. (`request=None` previews the configured
  request.) Backs "if I ask to load `xxx`, what will I get?"
- `available_databases()` — **discovery**: scan cwd / `user_dir` / `default_dir`
  and return the `.agg` files present (name + directory + path), as a DataFrame.
  "What *could* I load?", distinct from "what *would* this request load".

### D5 — one preprocessing owner
Move all text normalisation into `UnderwritingLexer.preprocess`. Verify the
lexer's continuation folding covers the 2–3-space indent case that
`read_database`'s `^  +`→tab pass was compensating for; if so, delete the
`read_database` regexes entirely. If the global `' +'`→`' '` collapse is load-
bearing for any `.agg` file, fold an equivalent, scoped step into the lexer
(not inside `[ ]`).

### D6 — explicit knowledge mutator
Add a small `add_entry(kind, name, spec, program)` (what `_interpret_program`
already does inline) and point the test fixtures at it instead of
`knowledge.loc[...] =`. This removes the only external dependency on the store
being a writable DataFrame and unblocks D1.

### D7 — consistent error policy
- A **literal** file named in an explicit `load(path)` call that is **missing**
  **raises** `FileNotFoundError` (the caller named it; a typo should be loud).
- The **configured** load (`load()` with no argument, incl. the lazy path) and
  any **glob** that matches nothing **warn** (a missing optional user dir, or a
  glob with no hits, is not an error — load-what's-there).
- Document the search path (cwd → `user_dir` → `default_dir`) and the `.agg`
  suffixing rule in one place.

## Decisions (confirmed with author, 2026-06-06)
- **Store is dict-backed** (`{(kind,name): ParsedProgram}`), DataFrame built on
  demand for `knowledge`/`discover` (D1).
- **Glob = union across dirs, cwd included** (D2).
- **Discovery** is in scope: `resolve_databases()` (preview a request) and
  `available_databases()` (what's on disk) (D4).
- **`'site'` token deleted** outright — functionality removed (D2).
- **`databases` reports the loaded files** (with paths); the request lives in a
  private `self._request`; no separate `loaded_files` attribute (D4).
- **All on `Underwriter`** — no separate `KnowledgeBase` class (D5/scope).
- **Rename outright** — no deprecated `read_database(s)` aliases (D4).
- **`reload()` = reset to as-created** (D4): clear the store, `databases`, and
  `_loaded`, restore the original `self._request`, then re-resolve and re-read.
  Drops any ad-hoc `load(...)`-ed files and any in-session `build(...)` entries —
  the object returns to exactly its constructor-time state.
- **Provenance + save are in scope** (D8): each entry carries a `source` tag
  (originating `Path`, or `'session'`); `to_agg(path, pattern='.*', kind='all',
  source='session')` exports matching entries to a `.agg` file, written to the
  user dir unless the path is absolute. Done in this plan ("all at once"), not
  deferred.

## The conceptual model (the "no active database" point)
Confirmed with the author, and worth stating because it shapes the save
question:

- The knowledge base is a **flat in-memory union**, `(kind, name) →
  ParsedProgram`, fed by the loaded `.agg` files **and** by in-session
  `build(...)` calls. There is **no "active" database.**
- Creating something new (`build('agg MyNew …')`) adds it to the union; it is
  **not** "saved into" any particular file and is **not** persisted — it lives
  only in this `Underwriter` until the process ends (or `reload()` drops it).
- `(kind, name)` is the unique key; a later entry with the same key overrides an
  earlier one (last load / last build wins).
- **Provenance (new):** tag each entry with its `source` — the `Path` it loaded
  from, or the sentinel `'session'` for in-session builds (the "current /
  unsaved" bucket). This answers "where did this come from?" (critique #7), lets
  `discover` show origin, and is what makes `to_agg` (below) work. Cheap: one
  extra field on `ParsedProgram`.

Note: the `Underwriter` docstring claims it can "persist DecL programs **to**
and from `.agg` files," but there is **no write/save method today** — only
reading. Either implement save (below) or correct the docstring.

### D8 — save: `to_agg` (pandas-style export)
Because there is no active database, "save" is an **explicit export of selected
entries to a named file**. Each entry already stores its `program` (DecL source)
line, so writing is just emitting those lines — and the `source` provenance tag
makes "save what I just built" the natural default.

`to_agg(path, pattern='.*', kind='all', source='session')` — write the matching
knowledge entries' `program` text to `path` (`.agg` suffix added if absent):

- **`pattern`** — regex matched against the entry *name* (same axis `discover`
  filters on). Default `'.*'` = all names.
- **`kind`** — filter by kind: `'all'` (default), `'agg'`, `'sev'`, `'port'`,
  `'distortion'` (also `'mvagg'`). Mirrors `discover(kind=...)`.
- **`source`** — provenance filter. Default **`'session'`** (the common case:
  the things you built this session that aren't in any file yet). `'all'` (or
  `None`) ignores provenance and exports every match. A file stem/`Path` exports
  only entries that came from that file (re-export / round-trip).
- **Write target:** an **absolute** `path` is used as given; otherwise the file
  is written to the **user dir** (`~/.aggregate`), so a saved book is
  immediately discoverable by `available_databases()` and re-loadable by name
  (`'user'` / `'all'`). `.agg` suffix added if absent.
- Writes one `program` line per entry, with a short comment header (`# written
  by aggregate <version> on <date>`). The result **re-loads cleanly** via
  `load(path)` — round-trip is the correctness contract.
- Returns the `Path` written (pandas `to_*` convention).

Default call `uw.to_agg('mybook')` therefore writes every in-session-built entry
to `~/.aggregate/mybook.agg`, ready to re-load by name later.

This also lets us **correct the docstring** — persistence "to and from" becomes
true rather than aspirational.

## Phases
- **Phase 1 — store + mutator (D1, D6).** Dict store; `source` provenance field;
  `add_entry`; DataFrame view; repoint test writers. Behaviour identical.
- **Phase 2 — loading pipeline (D2–D4, D7).** One resolver; `_loaded` flag;
  `load(request=None)` / `reload` / `resolve_databases` / `available_databases`;
  `databases` = loaded paths; error policy.
- **Phase 3 — preprocessing (D5).** Collapse the two layers into the lexer;
  delete the `read_database` regexes.
- **Phase 4 — persistence (D8).** `to_agg(path, pattern, source)` export; fix
  the over-claiming `Underwriter` docstring. Depends only on the Phase 1 `source`
  tag.

(All four are small and could land together; split only if review wants it.)

## Files
- `src/aggregate/underwriter.py` — the dict store + `ParsedProgram.source`,
  resolver, `load` / `reload` / `resolve_databases` / `available_databases` /
  `to_agg`, `__getitem__` / `_safe_lookup`, `_interpret_program`, `__repr__`
  (+ provenance), `_refresh_default_underwriter`, and the corrected class
  docstring.
- `src/aggregate/parser.py` — `UnderwritingLexer.preprocess` if D5 folds the
  munging in.
- `tests/conftest.py`, `tests/capture_sly_snapshot.py` — switch the direct
  `knowledge.loc[...] =` writes to `add_entry` (D6).
- `tests/test_underwriter.py` — extend to pin the new surface (`load()`
  idempotent; `load('missing')` raises; glob/configured no-match warns;
  `resolve_databases` preview; `available_databases` discovery; `reload` resets
  to as-created; `databases` reports loaded paths; `source` provenance;
  `to_agg` round-trip).

## Verification
- **Same knowledge:** `build.knowledge` (loading `test_suite`) has the same
  `(kind,name)` index and specs before/after; `discover()` output unchanged.
- **Idempotent load:** `load()` twice does not double-count; `_loaded` honoured.
- **Loaded view:** `uw.databases` lists the resolved `Path`s actually read;
  `resolve_databases(req)` matches what a subsequent `load(req)` reads.
- **Reload refreshes from disk:** editing a loaded `.agg` then `reload()` picks
  up the change.
- **Error policy:** `load('nope')` raises `FileNotFoundError`; a glob (or the
  configured set) with no matches warns, does not raise.
- **Save round-trip:** `build(...)` some entries → `to_agg(tmp)` →
  fresh `Underwriter().load(tmp)` reproduces the same `(kind,name)` specs;
  `source='session'` exports only in-session builds.
- **Preprocessing parity:** every bundled `.agg` (esp. the tabbed Portfolio
  layouts in `test_suite.agg`) parses identically after D5.
- `uv run pytest` green; `uv run ruff check src` clean. Version bump per the
  standing rule (plan-based code change).

## Notes
- Keep the import-time `build` cheap: no eager parse of `test_suite.agg`.
- `config.build.databases` (default `('test_suite',)`) is unchanged; this plan
  only rationalises how that request is *resolved and read*.
