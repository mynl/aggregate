---
name: version-bump
description: Bump the aggregate 1.0.0a version and commit it as one coherent
  unit: the gate run, pyproject.toml, the CHANGELOG section, dev/TODO.md, any
  regen the change implies, the finished plan moved to dev/done/, and a one-line
  commit message. Use when a plan-based change is finished, when asked to bump,
  ship, cut a version or commit a version, and to decide whether a given change
  bumps at all.
argument-hint: [Descriptive-Label]
---

# Version bump, aggregate (LIB)

House rules, followed without being re-asked. The point of the discipline is a
**bisectable** history: one bump, one commit, never batched, never deferred.

## 1. Does this bump at all?

- **Plan-based code change bumps.** Anything executed from a `dev/plan-*.md`, or
  any multi-step feature or refactor.
- **Pure tidying does not.** File moves, comment or doc-only edits, anything with
  no behavior change.

If it does not bump, **leave it uncommitted for the author** and say so. Do not
sweep non-bump work into a bump commit to tidy the tree. That is the single most
common way this goes wrong.

## 2. Read git state, never assert it from memory

```
git -C T:/worktrees/aggregate_REFACTOR status
git -C T:/worktrees/aggregate_REFACTOR log --oneline -5
```

The author commits frequently without announcing it, so any claim about what is
or is not committed written from session memory is routinely wrong by the time
it is read. Run these first. Mention uncommitted work only when the tree really
is dirty in a way that matters, for example an expected commit is missing. When
in doubt, stay silent about commits rather than nag.

Note what is already staged or modified that is **not** part of this bump. That
is the author's, and it stays behind.

## 3. Run the gate

Tier 3, everything, once, at this boundary:

```
UV_PROJECT_ENVIRONMENT=.venv uv run pytest -m 'slow or not slow'
```

If the change touches numerics, also run the `RuntimeWarning` gate. See the
`test-tiers` skill for both, and for why the environment variable is there.

**Green gate or no bump.** If it fails, report the failure with its output and
stop. Do not bump over a red suite and do not describe a red suite as mostly
green.

## 4. Assemble the unit

A version-bump commit is **one coherent unit**. It carries every item below that
the change implies, and never splits them across commits:

- The code change itself, with its tests.
- `pyproject.toml`, the `version = "1.0.0a*"` line. Read the current value, do
  not assume it. Recent history: `1.0.0a305`.
- `CHANGELOG.md`, a new `## <version>` section at the top of the entries, below
  the standing "API stability" preamble and above the previous entry.
- `dev/TODO.md`, tick the tracked item, noting the version and the
  `dev/done/plan-*.md` it landed as. If scope shifted, edit the entry rather
  than leaving it stale.
- `dev/FEATURES.csv` if the change touches the class by capability matrix,
  regenerated with `uv run python dev/regen_features.py`.
- The grammar snapshot if the grammar or transformer moved,
  `uv run python tests/capture_spec_snapshot.py`, with its diff read
  deliberately before it is committed.
- The finished plan moved from `dev/` to `dev/done/`.

**Never let a version land in a different commit from its CHANGELOG section.**
That is precisely what breaks bisect.

Watch for symlinked plans. `dev/done/plan-3d-plot.md` and
`dev/done/plan-pricing-natural-allocation.md` are symlinks whose canonical
copies live in the API repo, both retired to `done/` on 2026-08-21. Do not move
a symlink into `dev/done/` on its own; a shared plan is retired from the side
that owns it and the link follows in the same breath. The live LIB half of the
joint surface is `dev/plan-3d-plot-LIB.md`, a real file, and it moves to
`dev/done/` on its own when its two bumps land.

## 5. Write the CHANGELOG section

`CHANGELOG.md` **is** the full commit message. It is the running release-notes
draft, so write it for a reader who was not in the session: what landed, why,
and any breaking change called out plainly. A correctness fix that changes
numbers says so; a number that was wrong is not an interface to be preserved.

Add it at the close of the iteration. Do not defer it.

House style applies here as everywhere. **No dashes as punctuation, ever.**
Rewrite with a comma, a colon, parentheses, or a new sentence. Descriptive
bracketed labels like `[Chart-IR]`, never terse codes. US spelling.

`README.md` is the stable-audience front page and points at `CHANGELOG.md`.
Touch it only when that front-page material itself changes.

## 6. Commit, one line

```
[Descriptive-Label] a306: terse summary of what landed
```

Subject only. **No body, no trailers, no `Co-Authored-By`.** The CHANGELOG
section is the description; the one-liner is the index into it. Real examples
from this history:

```
[Ledger-Insurer-Abbreviated] a305: the insurer ledger narrows to EX, SD, CV and the adverse tail state
[Cede-Contra-Expense] a304: ceding commission folds into E as contra expense and the C column goes
[Session-Isolation] a302: one recipe base many users, Underwriter.fork() and preview() with RecipeNotFound
```

If the summary will not fit on one line, either the commit is too big or the
CHANGELOG entry is not doing its job. Fix the cause, do not wrap the subject.

The label is a descriptive `[Kebab-Case]` phrase naming the change. If one was
passed as `$ARGUMENTS`, use it.

Stage deliberately, by path. Do not `git add -A` over a tree that holds the
author's uncommitted work.

## 7. Never

- **Never push.** Pushing is the author's, on explicit request only.
- **Never `--no-verify`**, and never `--amend` anything already pushed. Prefer a
  new commit to rewriting one.
- **Never batch two bumps into one commit**, and never defer a bump to the end
  of a multi-step implementation. Three bumps leave three commits.
- **Never commit the author's unrelated work** along with the bump.

## 8. Report

State the new version, the gate tier that ran and its result, what went into the
commit, and anything deliberately left uncommitted for the author.
