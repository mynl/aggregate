---
name: execute-plan
description: Review a dev plan and, if it is sound, execute it end to end with
  house hygiene: phase by phase against the testmon loop, the tier 3 gate, one
  version bump commit per phase, execution notes recording every divergence,
  dev/TODO.md, dev-files.md, and the finished plan retired to dev/done/. If the
  plan is not sound, stop and ask in one batch rather than guess. Use when asked
  to review a plan and execute it if it is ok, to run, work, do or implement a
  dev/plan-*.md, or to pick up a plan that is specified and ruled but not yet
  executed.
argument-hint: dev/plan-<name>.md [review]
---

# Review and execute a plan, aggregate (LIB)

The standing request this answers: *review `plan/xxx`, if it is ok execute with
the usual hygiene, else ask questions.* Everything below is the usual hygiene,
written out so it never has to be re-asked.

Two skills carry pieces of this and are **called**, not restated. `test-tiers`
owns every pytest invocation. `version-bump` owns the gate, the CHANGELOG and
the commit. Read each when its moment arrives rather than working from a
recollection of what it says.

**Review only mode.** If `$ARGUMENTS` ends with `review`, stop after section 4
and report the review. No source file is touched in that mode.

## 1. Resolve the plan, and everything attached to it

The argument may arrive as `plan/xxx`, `dev/plan-xxx.md`, a bare stem, or a
bracketed label. Find it in this order:

```
ls dev/plan-*.md
ls dev/deferred/
ls dev/done/ | grep -i <stem>
```

- **Already in `dev/done/`.** The plan is retired. Do not re-execute it. Report
  what it says and ask what is actually wanted.
- **In `dev/deferred/`.** Parked past the beta on purpose. Ask before reviving.

Then read everything the plan is attached to, because a plan is rarely alone:

- **Is it a symlink?** `ls -l dev/<file>`, or a `120000` mode from
  `git ls-files -s dev/<file>`. Shared plans are canonical in the API repo with
  a link here (`plan-3d-plot.md` and `plan-pricing-natural-allocation.md` were
  both this shape). Read the canonical copy at
  `T:/worktrees/aggregate_api/dev/<file>`.
- **Is there a paired half?** Two arrangements are in use. The symlink, one file
  seen from both repos. And the reflection, a copy in each repo stating its own
  side as requirements (`plan-2d-punchup-requirements.md` here against
  `plan-2d-punchups.md` there). Read the other side either way. It holds the
  evidence and the acceptance criteria this side is written against.
- **Is there an existing execution notes file?** `dev/plan-<stem>-LIB.md`. If it
  exists, part of this plan has already run. Read its divergences before
  touching anything.
- **The control list.** `T:/worktrees/dev-files.md` carries a row per plan with
  its real status, and its "Getting to 1.0" section is the author's master list.
  The row is more current than the plan's own header.
- **The tracked item.** `dev/TODO.md`, the matching `[Bracket-Label]` entry.
- **Anything the plan cites**, including a
  `dev/note-from-aggregate-api-round-N.md` and the CHANGELOG entries that
  answered it.

## 2. Read the ground truth, never assert it from memory

```
git -C T:/worktrees/aggregate_REFACTOR status
git -C T:/worktrees/aggregate_REFACTOR log --oneline -8
grep -n '^version' pyproject.toml
```

The author commits frequently without announcing it, and a parallel session in
another worktree may be holding the next version number. Note what is already
modified in the tree that is **not** part of this plan. That is the author's and
it stays behind, uncommitted, at every bump.

## 3. The review, and what "ok" means

Work down this list. Each item is a real gate, not a formality.

**Status.** Plans carry a status line: *specified, ruled, not executed*, or
*OPEN, draft*, or *blocked on the author for section 4.2.1*. Anything short of
ruled and ready to execute is a stop.

**Open questions dispositioned.** A plan that asks the author something and
records no ruling is not executable. Ruled looks like a dated sentence saying
the numbers below are rulings rather than proposals.

**Order of work.** Cross-repo plans state it, and it is usually LIB first and in
full, then the API runs `uv sync --extra dev` and starts its half. If this repo
is second in line and the other side has not landed, stop.

**The plan's facts against today's code.** A plan names files, line numbers,
symbols and the version it was written against. Several bumps may have landed
since. Verify every named path exists, every symbol still spells that way, and
every quoted snippet still reads that way. **A plan claim the code contradicts
is a blocker, not a detail:** the ruling may have been made on a premise that is
no longer true.

**Names vetted, at review time and not after.** For any new public method,
attribute or kwarg the plan introduces, `rg` it across `src/aggregate` for a
collision with an existing callable or attribute on the same class or its
parents. An instance attribute set in `__init__` silently shadows a method of
the same name, which went unnoticed through an entire plan once
(`dev/done/plan-approximate.md`). Prefer a noun for a stored value and a verb
for an action. Say the chosen names in the review so they can be challenged.

**Acceptance criteria.** The plan should name what to measure. If it has no
verification section, ask for one rather than inventing a standard for it.

**Blast radius.** What else reads what this changes. `rg` the symbols. If the
grammar or the transformer moves, the snapshot regenerates and its diff must be
read deliberately. If a public surface moves, `dev/FEATURES.csv` regenerates. If
a public name the API imports moves, that is a ripple, not a local change.

**Ripple to the app.** DecL grammar or keyword changes must be checked against
`T:/worktrees/aggregate_api/web/src/decl-keywords.json`, which mirrors
`parser_errors._TERMINAL_LABELS` by hand. Exhibit and chart registry changes,
and any capability flag, are the app's business too. State the ripple in the
review even when nothing on this side has to change.

## 4. The fork

**Stop and ask** on any of these:

- an undispositioned open question, or a status that is not ruled and ready
- a plan claim the current code contradicts
- a name collision, or a name that reads wrong against the existing surface
- an unsatisfied cross-repo dependency, or an unsynced sibling
- a missing or unmeasurable acceptance criterion
- work the change cannot land without and the plan does not cover

Ask **all** the questions in one batch, and then **wait**. The author
multiprocesses and will answer. Do not proceed on a provisional pick after a
silence, and do not start the easy phases while a blocker is outstanding unless
the plan's own phase order makes them genuinely independent.

**Everything smaller is executed and recorded as a divergence**, never silently
absorbed. That is the house convention:
`dev/done/plan-pricing-exhibits-LIB.md` section 3 carries nine of them, each
saying what the plan specified, what the code does instead, and why. A recorded
divergence is a good outcome. An unrecorded one is the failure.

## 5. Execute

- **Phase by phase, in the plan's order. One phase, one bump, one commit.**
  Never batch two phases into one commit, and never defer a bump to the end.
- **The edit loop is tier 1** from `test-tiers`, the testmon invocation with all
  its load-bearing flags. Do not run the full suite inside the loop.
- **Write the tests the plan asks for.** Any new DecL program used in a pytest
  case is appended to `src/aggregate/agg/decl-testers.agg` under the matching
  section, and it must round-trip.
- **Docstrings on everything new or modified.** NumPy style, and for
  non-trivial mathematics the why goes in Notes. This is an actuarial library;
  the why is often the point.
- **House style in everything authored.** No dashes as punctuation, ever, in
  code comments, docstrings, CHANGELOG, plans or replies. US spelling.
  Descriptive `[Bracket-Label]` names, never terse codes. One paragraph is one
  physical line in `.md` and `.rst` prose.
- **Keep the execution notes open as you go.** `dev/plan-<stem>-LIB.md` for a
  shared or reflected plan, or an execution log section appended to a plan that
  is LIB only. Record each divergence at the moment you make it. Reconstructing
  them at the end is how they get lost.

## 6. Hygiene at each bump

Call the `version-bump` skill and follow it. It owns the tier 3 gate, the
`pyproject.toml` line, the CHANGELOG section, `dev/TODO.md`, the regens, the
one-line commit and the never-push rule.

Three things this skill adds on top of it:

- **The execution notes file** ships in the same commit as the phase it
  describes.
- **The `dev-files.md` row** at `T:/worktrees/dev-files.md` gets the plan's new
  status and version range. It is the control list the next session reads first.
- **A numerics-touching phase also runs the `RuntimeWarning` gate**, per
  `test-tiers`.

## 7. Retire the plan

In the final phase's commit, and not before:

- Move the plan from `dev/` to `dev/done/`, together with its execution notes.
- **Symlink care.** A shared plan is retired from the side that owns it and the
  link follows in the same breath. Do not move a symlink into `dev/done/` on its
  own; that dangles it. A reflected plan is retired on each side independently,
  when that side's half lands.
- Tick the `dev/TODO.md` entry with the version and the `dev/done/plan-*.md` it
  landed as. If scope shifted, edit the entry rather than leaving it stale.
- Update the `dev-files.md` row to DONE with the version range.

## 8. Report

- The version range, and what landed in each bump.
- Which test tiers ran, with the actual commands, and their results. A failure
  is reported with its output, never summarized as mostly green.
- **Every divergence**, with a pointer to where it is recorded.
- Anything left for the author: an open question, a ripple the app owes, a
  finding the plan did not anticipate.
- What in the tree was deliberately left uncommitted.
