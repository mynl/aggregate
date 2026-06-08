# Plan / report: Hygiene batch 1 — H1, H2, H3, H6

(Requested as `plan-hygene-1.md`; spelled `hygiene` to match Track **H — Hygiene**.)

Scope: investigate and recommend on TODO items **H1** (relocate
`make_ceder_netter`), **H2** (dedupe var/tvar), **H3** (import-dependency audit),
**H6** (mark done). **H4, H5 deliberately untouched this round.**

---

## H6 — Underwriter database-loading rewrite → mark done ✅

Already marked done in `dev/TODO.md` (status table line 49, detail line 173,
`dev/done/plan-databases.md`). One refinement per author: it landed across **two**
versions, not one.

- **a32** — the loading rewrite itself (dict-backed store + DataFrame view,
  `source` provenance, glob-aware resolver). `dev/done/plan-databases.md`.
- **a35** — the default changed so a fresh `Underwriter` loads **no** databases
  rather than `default` (commit `637febca`).

**Action:** edit the H6 detail line to read "done in **1.0.0a32** + **1.0.0a35**"
and add the a35 clause. No code. Trivial bookkeeping.

---

## H2 — Dedupe var/tvar → **non-issue, confirmed; close it**

The author's hunch is correct: **there is no duplicated implementation.**

- `utilities.make_var_tvar(ser)` (`utilities.py:541`) is the **single** core
  implementation — builds the upper/lower quantile and TVaR functions from a
  probability series.
- `Aggregate._make_var_tvar` (`distributions.py:6263`) and
  `Portfolio._make_var_tvar` (`portfolio.py:1471`) are **thin per-instance
  wrappers** that just call `make_var_tvar(ser)` and stash the three resulting
  callables. They differ only in *where* they put the result: `Aggregate` returns
  a dict (it has both a severity and an aggregate function to manage);
  `Portfolio` writes straight to `self._var_tvar_function` (only one). `bounds.py`
  also calls the utility directly.

So the var/tvar math lives in exactly one place; the methods are object-specific
glue, not duplication. The TODO's framing ("utilities.make_var_tvar vs
distributions._make_var_tvar") reads as two implementations but they are
caller/callee.

**Action:** close H2 as already-deduped (no work). Optional micro-tidy (not worth
it): the two wrappers' three-line `{upper,lower,tvar}` assembly could share a
helper, but they store results differently — leave them.

### Incidental (out of scope — flag only)

While confirming, I noticed a likely copy-paste bug in
`Aggregate.tvar_sev` (`distributions.py:6293-6297`): it tests
`self._var_tvar_function is None` but then assigns `self._sev_var_tvar_function`
and returns `self._var_tvar_function['tvar']` — the `sev`/non-`sev` members look
crossed. **Not part of H2;** logging here so it isn't lost — worth a separate
look.

---

## H1 — Relocate `make_ceder_netter` → **do it; small, contained**

`make_ceder_netter` (`utilities.py:319`) is reinsurance-ceding logic used **only
by `distributions.py`** (the occ/agg reinsurance application at lines 2791, 2842,
5014). Confirmed no use in `portfolio.py`, `bounds.py`, or elsewhere in `src`. It
belongs with its single consumer.

**Edits:**

| File | Change |
|---|---|
| `utilities.py` | remove the `def make_ceder_netter` (and its docstring), and drop `'make_ceder_netter'` from `__all__` (line 27) |
| `distributions.py` | define `make_ceder_netter` here; remove it from the `from .utilities import (… make_ceder_netter …)` line (43) |
| `tests/test_reins_buckets.py:23` | **split** the import: `from aggregate.distributions import make_ceder_netter` **and** keep `from aggregate.utilities import _validate_reins_layers` |

**Dependencies of the function:** numpy (`np.inf`, array build) and a piecewise-
linear interpolation (scipy `interp1d`). `distributions.py` already imports numpy
and `from scipy import interpolate` — confirm the exact `interp1d` reference on
the move (import `from scipy.interpolate import interp1d` if the body uses the
bare name). No new third-party imports.

**Sibling `_validate_reins_layers` — LEAVE in `utilities.py`** (author decision):
it is genuinely a utility and is only called by test functions, so it stays put.
Only `make_ceder_netter` moves; the test import is *split* (see table above).

**No deprecation re-export** — hard move, consistent with the project's
no-alias rename style (these are internal/`__all__`-listed helpers, alpha).

---

## H3 — Import-dependency audit → **three unused deps to drop; two to consider lazy**

The runtime dependency list (`pyproject.toml:29-40`) carries small packages that
are **declared but never imported** in `src/aggregate`:

| Dep | Imported in `src`? | Recommendation |
|---|---|---|
| **`cycler`** | **No** (grep: zero hits; `bounds.py` already uses stdlib `itertools.cycle`) | **Remove.** This is the author's remembered case — a non-standard dep with the stdlib replacement already in use. matplotlib pulls cycler transitively anyway. |
| **`psutil`** | **No** (zero hits) | **Remove.** Nothing reads system/memory info. |
| **`ipykernel`** | **No** (zero hits) | **Remove from runtime deps.** A kernel is not a library dependency; at most it belongs in the `notebook` extra. |
| **`jinja2`** | **No direct import** | **Remove.** Author confirms it was only for book-example styling, logic long gone — obsolete. |
| `IPython` | Yes — `utilities.py:14` (`from IPython.display import HTML, Markdown, display`), top-level | **Keep, but make lazy** (see below). |
| `Pygments` | Yes — `utilities.py:8-10`, `decl_pygments.py`, top-level | **Keep, but make lazy** (see below). |
| `lark`, `matplotlib`, `numpy`, `pandas`, `scipy` | Yes | Core. Keep. |

**Primary action:** drop `cycler`, `psutil`, `ipykernel`, **and `jinja2`** from
`dependencies` — all four are unused. Low-risk (removing things nothing imports).

**Also in scope — lazy imports (author: library load time is too long):**
`IPython` and `Pygments` are imported **eagerly at the top of the core
`utilities.py`**, so every `import aggregate` drags both in. Convert them to
**lazy imports inside the functions that use them** (the `qd`/display helpers for
`IPython.display`; the DecL highlighting helpers for `pygments`). `utilities.py`
is imported by `distributions`/`portfolio`/everything, so this directly shortens
the cold `import aggregate` path. Verify the only `IPython`/`pygments` uses are
inside a handful of functions (display + highlight) so the top-level imports can
be removed cleanly; keep a module-level comment noting the deliberate deferral.
*(Note: this is an import-time optimization, not a dep removal — IPython and
Pygments remain declared dependencies.)*

---

## Verification

- **`uv run pytest`** — full suite. Specifically exercises:
  - `tests/test_reins_buckets.py` after the H1 import change (the one test that
    imports `make_ceder_netter` directly).
  - everything that builds reinsurance aggregates (H1 is a pure relocation — no
    behaviour change; freeze/check would be all-match but pytest already covers
    the reinsurance paths).
- **`python -c "import aggregate; from aggregate import build; build('agg X dfreq[1] dsev[1]')"`**
  after the H3 dep removals — confirms nothing relied on `cycler`/`psutil`/
  `ipykernel`/`jinja2` at import or build time.
- **Lazy-import checks:** after deferring `IPython`/`Pygments`, confirm (a) the
  display helpers (`qd` etc.) and the DecL HTML highlighting still work, and
  (b) `import aggregate` no longer imports IPython/pygments eagerly — e.g.
  `python -c "import sys, aggregate; assert 'IPython' not in sys.modules and 'pygments' not in sys.modules"`.
  A before/after `python -X importtime -c "import aggregate"` is a nice optional
  confirmation of the load-time win.
- No numeric change anywhere → `freeze_knowledge` not required, but harmless.

## Housekeeping (standing rules)

- Bump `pyproject.toml` `1.0.0a*` — H1 (code move), H3 (dependency change +
  lazy imports) are real changes; H2/H6 are doc-only and ride along.
- `CHANGELOG.md`: "Hygiene: relocated `make_ceder_netter` to `distributions`;
  dropped unused runtime deps `cycler`, `psutil`, `ipykernel`, `jinja2`;
  deferred `IPython`/`Pygments` to lazy imports to cut `import aggregate` time."
- `dev/TODO.md`: mark **H1**, **H3** done with the version; **H2** done as
  non-issue; refine **H6** with the a35 note.
- Move this file to `dev/done/plan-hygiene-1.md` on landing.

## Resolved (author decisions)

1. **H1 sibling:** leave `_validate_reins_layers` in `utilities.py` (it's a real
   utility, only called by tests). Move **only** `make_ceder_netter`; split the
   test import.
2. **H3 `jinja2`:** remove — obsolete book-example styling, gone.
3. **H3 lazy imports:** in scope — defer `IPython`/`Pygments` to shorten load
   time.

## Out of scope

- **H4** (NumPy docstring sweep) and **H5** (`pedagogy` figure migrations) — not
  this round.
- The incidental `tvar_sev` member-crossing bug (flagged under H2) — separate.

---

## Execution notes (landed 1.0.0a44)

Two adjustments to the plan-as-written, made during execution:

1. **Pygments deferral dropped (author decision).** Investigation showed the
   plan's lazy-Pygments goal was unachievable as scoped and worthless anyway:
   `__init__.py` eagerly imports `decl_pygments`, which *must* import pygments to
   define its `RegexLexer` subclass (and `pyproject.toml` registers it as a
   pygments entry point), so pygments is always loaded by `import aggregate`
   regardless of `utilities.py`. Measured cost: `IPython.display` ~900 ms vs
   pygments ~0 ms. Per author, **only IPython was deferred**; Pygments stays a
   top-level import and a declared dep. The verification assertion was narrowed
   to `'IPython' not in sys.modules`.
2. **`_validate_reins_layers` moved to `distributions.py` alongside
   `make_ceder_netter`.** The plan's H1 originally left the validator in
   `utilities`, but it is *only* called by `make_ceder_netter` (plus the tests),
   so the author decided it should travel with its sole consumer. Both functions
   now live in `distributions`; the test imports both from there. Also removed
   the now-unused `from scipy.interpolate import interp1d` from `utilities.py`
   (the moved `make_ceder_netter` uses `interpolate.interp1d` in its new home,
   matching the `distributions.py` idiom).

Verification: `import aggregate` confirms IPython deferred / pygments present;
reins build + relocated `make_ceder_netter` exercised directly; lazy display
branch (`decl_pprint(html=True)`) works on use. **Full suite: 1091 passed.**
Incidental `tvar_sev` bug filed as **B4** in `dev/TODO.md`.
