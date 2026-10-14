# Plan: make `Underwriter.load()` and `fork()` thread safe

Patch bump (1.5.0 to 1.5.1). Raised from `aggregate_api`, where it surfaced as
a production bug; that repo's `dev/plan-warm-library-at-startup.md` works
around it on the api side and lands first. This plan fixes the cause here.

## Goal

No caller, on any thread, can observe or fork a half-read recipe base. A
thread that asks for the base while another thread is reading it waits for the
read to finish.

## Current behavior

`Underwriter.load()`, the `request is None` branch (`underwriter.py`, about line
975):

```python
if self._loaded:
    return []
paths = self._resolve_database_request(self._request, explicit=False)
self._loaded = True          # set BEFORE the files are read
...
for p in paths:
    if self._read_file(p): ...
```

There is no lock. The flag is set first for a reason that must be kept:
reading `library.agg` interprets each statement, a `sev.X` / `agg.X` reference
goes through `_safe_lookup`, then `recipe()`, which begins `if not
self._loaded: self.load()`. Setting the flag first is what stops that
same-thread re-entry from reading the file again, recursively.

The cost of the ordering is a cross-thread hole. While thread A is reading
(about three seconds for `library.agg`), thread B sees `_loaded` True and
proceeds against a partly filled `_recipes`:

- `recipe()`, `recipes`, the `.agg` writer and the parser callback all guard
  with `if not self._loaded: self.load()` (around lines 2221, 2276, 3505), so
  B gets `RecipeNotFound` for an entry A has not reached yet.
- `fork()` guards the same way (around line 1137) and then copies the dict, so
  B's fork is a **permanent** snapshot of the partial base. That is the bad
  one: the fork never catches up, by design ("a fork is a snapshot, not a
  subscription").

`build` is constructed unloaded (`build._loaded` is False after `from
aggregate import build`), so the first touch in any process is a load. The
defect is in released 1.0.1 too.

How it bit, in `aggregate_api`: on a freshly started server the web app fires
two requests at once; one starts the read, the other forks a session base off
it mid-read. Reproduced in a fresh process: the base ends with 198 recipes and
the fork with 113, and `Capstone.ExposureRating` then fails with no recipe
named `Capstone.Gross` of kind `agg`. Any multi-threaded host (a web server, a
notebook running a thread pool, Dask threads) can hit it.

`reload()` and `_refresh_default_underwriter()` clear `_recipes` and reset the
flag with no lock either, so a concurrent reader can see an empty base.

## Change

**[load lock]** Give every `Underwriter` a `threading.RLock`,
`self._load_lock`, created in `__init__`. Reentrant, because the same thread
re-enters `load()` through the lookup path above.

**[two flags]** Split "read has finished" from "read is under way":

- `_loaded` is set only **after** the configured request has been read. It
  keeps its meaning for every existing reader (`__repr__`, the guards).
- `_loading`, new, is True while the configured read runs. It replaces the
  early flag as the same-thread re-entry guard.

```python
def load(self, request=None):
    with self._load_lock:
        if request is None:
            if self._loaded or self._loading:
                return []
            self._loading = True
            try:
                paths = self._resolve_database_request(self._request, explicit=False)
                read = [p for p in paths if self._read_file(p)]
            finally:
                self._loading = False
            self._loaded = True
            return read
        paths = self._resolve_database_request(request, explicit=True)
        return [p for p in paths if self._read_file(p)]
```

Why this closes the hole without touching the guards: a thread that finds
`_loaded` False calls `load()` and blocks on the lock until the reader is done,
then finds `_loaded` True and returns. The reading thread itself re-enters
through the `RLock` and returns on `_loading`. Nobody else can get past a guard
early, because `_loaded` is no longer True early.

**A decision for the author: what a failed read leaves behind.** Today a read
that raises part way (the unique-names check in `_check_library_names_unique`
is the realistic case) leaves `_loaded` True over a partial base, so the next
call quietly works against it. The sketch above leaves `_loaded` False, so the
next call retries and raises again. Recommendation: retry, since a partial base
that looks loaded is the very defect this plan removes. If retrying is wrong
for some caller, keep today's behavior by moving `self._loaded = True` into the
`finally`.

**[fork under the lock]** Wrap the body of `fork()` in `with
self._load_lock:` so the dict copy cannot interleave with an additive
`load(path)` on another thread. Give the fork its own lock and flags, beside
the other per-instance resets it already does (`_parser`, `_lexer`,
`_sev_ref_stack`): `forked._load_lock = threading.RLock()`, `forked._loading =
False`. A shallow copy would otherwise share the parent's lock, which is
harmless but says the wrong thing.

**[reset under the lock]** `reload()` and `_refresh_default_underwriter()`
take the same lock around their reset, and `reload()` calls `load()` inside it
(reentrant), so no reader sees the empty base between the clear and the
re-read.

**Check before landing:** that nothing copies, deep-copies or pickles an
`Underwriter` other than `fork()`'s shallow copy. An `RLock` cannot be pickled
or deep-copied. A search of `src/aggregate` for `deepcopy(` / `pickle` over an
underwriter found nothing, but `joblib` or multiprocessing in a notebook would
pickle `build` if anyone passed it to a worker. If that is a supported use, add
`__getstate__` / `__setstate__` that drop and recreate the lock.

## Files

- `src/aggregate/underwriter.py`: `__init__`, `load`, `fork`, `reload`,
  `_refresh_default_underwriter`, and the `fork` / `load` docstrings (state the
  thread-safety guarantee, and why the lock is reentrant).
- `tests/test_underwriter.py`: the tests below.
- `CHANGELOG.md`, version files: patch bump 1.5.1.

## Tests

Deterministic, no sleeps-and-hope. Use a fresh `Underwriter` over
`library.agg`, never the `build` singleton, so the suite's shared base is not
disturbed.

- **[fork waits for the read]** Monkeypatch the instance's `_read_file` with a
  wrapper that sets a `threading.Event` once it has started and then waits on a
  second event before continuing. Thread A calls `load()`. When the first event
  fires, thread B calls `fork()`; release A. Assert B's fork holds every entry
  the finished base holds, and `('agg', 'Capstone.Gross')` among them.
- **[lookup waits for the read]** Same harness; thread B calls
  `recipe('Capstone.ExposureRating')` mid-read and gets the recipe, not
  `RecipeNotFound`.
- **[re-entry still works]** A plain single-threaded `load()` of `library.agg`
  succeeds; it exercises the `sev.X` / `agg.X` re-entry path, which an `RLock`
  plus `_loading` must not deadlock or recurse on.
- **[failed read]** Whichever rule the author picks above, pin it: a read that
  raises leaves `_loaded` as decided.

## Acceptance

- Full LIB suite green.
- The four tests above pass, and the two race tests fail on 1.5.0 (run them
  once against the unfixed code to prove they detect the defect).
- In `aggregate_api`, with this build installed editable and the api's
  startup read temporarily removed, the fresh-server case (restart, open
  `/?example=Capstone.ExposureRating` at once) still builds.

## Order with the api plan

The api plan lands first: it ships today against every released `aggregate`
in its pin (`>=1.0.1`), all of which carry this defect. This plan follows at
the library's pace. Once a release carrying it is the api's floor, the api's
startup read is no longer needed for correctness but is still worth keeping,
since it moves the three second read off the first request.
