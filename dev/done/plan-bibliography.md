# Plan — re-source the docs bibliography from the uber-library

> **Status: READY — not yet executed.** Doc-tooling change. Author requested a
> visible **version bump** (`1.0.0a89` → `1.0.0a90`) even though it is doc-only,
> "just so I can see it." Sphinx docs are **not** rebuilt in the iteration loop
> (CLAUDE.md) — the author rebuilds manually.
>
> **Executed in an isolated worktree, not on `REFACTOR` directly** — so the author
> can carry on with `REFACTOR` code work in parallel. See *Branch / worktree
> workflow* below.

## Branch / worktree workflow

This plan runs on its own branch in a sibling worktree and merges back when done.
The author keeps working on `REFACTOR` (in `T:\worktrees\aggregate_REFACTOR`)
throughout — the two barely overlap (this work is in `docs/`, the code refactor
is in `src/aggregate/`).

**Handshake (agreed):**

1. **Author** commits this plan on `REFACTOR`, then asks for execution.
2. **Claude** creates the branch + worktree and does *all* the work there:
   ```
   git worktree add -b DOCS ../aggregate_DOCS REFACTOR
   ```
   → branch `DOCS` off the current `REFACTOR` tip, in `T:\worktrees\aggregate_DOCS`.
   Then `uv sync` (with `UV_LINK_MODE=copy`) in that worktree once, before running
   the extractor. **All edits/steps below happen in `aggregate_DOCS`; this
   `REFACTOR` checkout is left untouched.**
3. **Claude** finishes, then **asks the author to review and commit** on `DOCS`.
4. **Author** asks Claude to merge back; **Claude** merges `DOCS` into `REFACTOR`
   (from the `REFACTOR` worktree: `git merge DOCS`), resolving the few small
   overlap conflicts (`pyproject.toml` version line, `CHANGELOG.md`,
   `dev/TODO.md`). The author does the final commit of the merge.

**Version coordination:** the `DOCS` branch bumps `a89 → a90`. If `REFACTOR` also
bumped to `a90` meanwhile, the merge collides on that one line — resolve by giving
one of them the next number. Flag the chosen numbers at merge time.

## Goal

Re-source every bibliography reference used in the Sphinx docs from the author's
master library, `C:/S/TELOS/Biblio/uber-library.bib` (~7,100 entries, a symlink
to the `archivum`-maintained `bibtex.bib`), replacing the two ad-hoc local files
`docs/extract.bib` (108 entries) and `docs/books.bib` (31 entries).

### End state (the contract)

1. **Every `:cite*:` role in `docs/**/*.rst` points at either** (a) a **canonical
   uber-library key**, or (b) one of the few keys owned by a new hand-maintained
   `docs/manual.bib`.
2. **A committed script regenerates `docs/extract.bib` from uber-library** on the
   author's machine. Because the regenerated `extract.bib` is committed, Read the
   Docs (Ubuntu, no access to `C:/S/...`) builds offline with every ref present.

### Hard constraint (why we cannot just repoint `conf.py` at uber)

`.readthedocs.yaml` builds on RTD's Ubuntu runners, which have **no access** to
`C:/S/TELOS/Biblio/uber-library.bib`. So `bibtex_bibfiles` must list **only
repo-local files**. "Re-source from uber" therefore means *extract the cited
entries into a committed local file*, not reference the external path.

---

## Current state (as surveyed)

- Engine: `sphinxcontrib.bibtex` (`docs/conf.py:91`), author-year style.
  `docs/conf.py:133` → `bibtex_bibfiles = ['extract.bib', 'books.bib']`.
- `docs/7_bibliography.rst` renders with `:cited:` — only cited entries appear,
  so uncited library entries are harmless ballast.
- **131 distinct cite keys** used across ~40 `.rst` files (233 `:cite*:` roles:
  218 `:cite:t:`, 14 `:cite:p:`, 1 `:cite:`).
- `docs/bib.bat` currently runs `python -m great.bib_hacker` (the prior, external
  extraction tool) — to be repointed at the new in-repo script.
- Version: `pyproject.toml` `1.0.0a89`.

### Key resolution (all 131 settled)

- **103 keys already match uber verbatim** → pulled as-is, no `.rst` edit.
- **Renames** — present in uber under a different key; rewrite the `.rst` cites:

  | local key | → canonical uber key | note |
  |---|---|---|
  | `billingsley` | `Billingsley2012` | later-year rule (vs `Billingsley1986`) |
  | `feller71` | `Feller1971` | |
  | `JKK` | `Johnson2005` | later-year rule (vs `Johnson1992`) |
  | `Consul1973` | `Consul1973a` | exact-title match |
  | `Marinacci2004b` | `Marinacci2004a` | |
  | `Sherris2006a` | `Sherris2006` | |
  | `WangS1998` | `Wang1998a` | |
  | `Luo2010` | `Luo2011` | local key year was wrong |
  | `Svindland2009` | `Svindland2010a` | |
  | `Strang1986am` | `Strang1986` | |
  | `Stein1971bk` | `Stein1971` | |
  | `Stein2011bk` | `Stein2003a` | same book, 2003 Princeton ed. |
  | `Loeve2017` | `Loeve1955` | merges with existing `Loeve1955` cite |
  | `Grandell1997` | `Grandell1997` | author corrected the uber year typo (1977→1997) during execution; key unchanged |
  | `Daykin1993` | `Daykin1994` | |
  | `Carter2013` | `Carter1979a` | only uber candidate (1979 ed.) |
  | `Malliavin1995bk` | `Malliavin2012` | |
  | `Korner2022` | `Korner1988` | |
  | `Mccullagh2019` | `McCullagh1989` | author confirmed (vs `McCullagh1989a`) |
  | `Fisher2019` | `Fisher2017` | |
  | `LM` | `Klugman2019` | *Loss Models* 5th ed. (author choice) |
  | `KPW` | `Klugman2019` | collapses with `LM`/`kpw5`; **was cited-but-undefined → now fixed** |
  | `kpw5` | `Klugman2019` | |
  | `PIR` | `Mildenhall2022a` | house alias; merges with existing cite |
  | `AonBenfield2015f` | `AonBenfield2015e` | *Insurance Risk Study* 2015 (author choice; **not** `Mildenhall2016` "First Ten Years") |

- **`manual.bib` (not in uber — keys kept as-is, no `.rst` change):**
  - `Bertram1983` — J. Bertram, *Calculations of aggregate claims distributions in
    case of negative risk sums*, 17th ASTIN Colloquium, Lindau, Germany, 1983.
  - `Panjer1992` — Panjer & Willmot, *Insurance Risk Models*, Society of Actuaries,
    Schaumburg IL, 1992.
  - `Lukacs1970bk` — Lukacs, *Characteristic Functions*, 1970.
  - `McKean2014bk` — McKean, *Probability: The Classical Limit Theorems*, 2014.
  - (carried forward, currently **uncited** so non-rendering, kept for future use:
    `Scipy`, `pandas`, `Python3`, `sly`, `Hunter2007` (matplotlib), `Levine1992`,
    `DOCS`, `Bertram1981`.)

> **Note for the author (optional, your call):** `Lukacs1970bk` and `McKean2014bk`
> are legitimate academic books absent from uber. They live in `manual.bib` for
> now; if you'd rather they be canonical, add them to uber via `archivum` and
> re-run the script — they'll move to `extract.bib` automatically and the
> `manual.bib` entries can be deleted.

---

## The update script — `docs/update_extract_bib.py`

Python (robust balanced-brace BibTeX parsing; PowerShell is awkward for this).
Run on the author's machine: `uv run python docs/update_extract_bib.py`.

**Algorithm**

1. Read all cite keys from `docs/**/*.rst` (`:cite:`, `:cite:t:`, `:cite:p:`,
   comma-separated lists inside one role).
2. Read `docs/manual.bib`, collect its keys → the **excluded set** (refs that are
   intentionally not in uber).
3. For each remaining key, locate `@type{key,` in `uber-library.bib` and capture
   the full entry by brace-balancing to the closing `}`.
4. Write `docs/extract.bib`: a provenance header (source path, generation note,
   "DO NOT EDIT BY HAND — regenerate via docs/update_extract_bib.py"), then
   entries sorted by key.
5. **Report and exit non-zero** if any cited key is neither found in uber nor
   owned by `manual.bib` (a genuinely broken citation) — so drift is caught
   before commit. Also warn on uber keys that resolve to 0 or >1 matches.

The uber path is read from a module constant (default
`C:/S/TELOS/Biblio/uber-library.bib`), overridable via `--uber <path>` /
env var, so it is never hard-wired into the RTD build path. The script **only
reads** uber-library (never writes it — CLAUDE.md standing order).

---

## File changes

- **New** `docs/update_extract_bib.py` — the extractor (above).
- **New** `docs/manual.bib` — hand-maintained non-uber + software entries.
- **New** `docs/README.md` — documents the workflow: *refs are sourced from the
  author's uber-library; after adding/altering citations, run the script on the
  author's machine, commit the regenerated `extract.bib`, and RTD builds offline.
  `manual.bib` is hand-edited for refs not in uber (and software). Never edit
  `extract.bib` by hand.*
- **Regenerate** `docs/extract.bib` — by running the script (overwrites the
  current hand-rolled file).
- **Edit** ~40 `docs/**/*.rst` — rewrite cite keys per the rename table
  (mechanical, exact-key find/replace; manual.bib keys unchanged).
- **Edit** `docs/conf.py:133` → `bibtex_bibfiles = ['extract.bib', 'manual.bib']`.
- **Edit** `docs/bib.bat` → call `uv run python docs/update_extract_bib.py`.
- **Delete** `docs/books.bib` (academic entries now flow from uber into
  `extract.bib`; non-uber/software entries moved to `manual.bib`).
- **Bump** `pyproject.toml` `1.0.0a89` → `1.0.0a90`.
- **Add** `CHANGELOG.md` `## 1.0.0a90` section (bibliography re-sourced from
  uber-library; new `update_extract_bib.py`; `books.bib` retired; key renames).
- **Update** `dev/TODO.md` (mark the bibliography task done, ref this plan) and
  move this plan to `dev/done/` when it lands.

---

## Execution order

0. **Create the worktree** (per *Branch / worktree workflow*):
   `git worktree add -b DOCS ../aggregate_DOCS REFACTOR`, then `uv sync`
   (`UV_LINK_MODE=copy`) in `aggregate_DOCS`. **All steps below run there.**
1. Write `docs/update_extract_bib.py`.
2. Write `docs/manual.bib` (Bertram, Panjer book, Lukacs, McKean + software
   carried from `books.bib`).
3. Rewrite the ~40 `.rst` cite keys per the rename table.
4. Run the script → regenerate `docs/extract.bib`; confirm it exits 0 (no
   unresolved cites). Spot-check a few pulled entries against the intended works.
5. Edit `conf.py`, `bib.bat`; delete `books.bib`.
6. Write `docs/README.md`.
7. Bump version; update `CHANGELOG.md`, `dev/TODO.md`.
8. **Ask the author to review and commit** on `DOCS` (do not commit).
9. On the author's request, **merge `DOCS` back into `REFACTOR`** (from the
   `REFACTOR` worktree).

Steps 1–2 do not depend on the rename answers and could go first; the order above
keeps the `.rst` rewrite (3) and the regenerate-and-verify (4) adjacent so the
script's "no unresolved cites" check validates the rewrite immediately.

---

## Risks / notes

- **Verbatim-key collision (low):** a shared `AuthorYYYY` key *could* point at a
  different work in uber than in the old `extract.bib`. The author maintains both
  libraries on the same convention, so risk is low; step 4 spot-checks.
- **`Grandell1977` year:** uber's only "Mixed Poisson Processes" candidate is
  keyed 1977 though the book is 1997 — flagged for a glance; mapping is otherwise
  unambiguous (sole author+title match).
- **No doc build here:** per CLAUDE.md the Sphinx build is slow and run manually
  by the author; this plan does not build docs. Note in the commit that docs need
  a rebuild.
- **`great.bib_hacker`** (the old external tool) is superseded by the in-repo
  script and no longer referenced after `bib.bat` is repointed.
