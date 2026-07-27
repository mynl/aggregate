# Getting to v1.0

Written 2026-07-26, at `1.0.0a148` on branch `REFACTOR`.

## Background
The idea the plan is built on:

* Programs that **use** (exercise) the API tell you whether it is right.
* Words that **describe** (explain) the API tell you nothing.
* Prose written about an unfrozen API is what gets thrown away. Programs written against an unfrozen API are what freezes it.
* So write the programs before you freeze it, and the words after.

**A note on docs**: docs blend code-generated contents (ch 3 and 4) with custom written examples and theory (ch 2 and 5). Plan calls for chs 2 and 5 material to be stripped out and placed in a separate Monograph document (to be submitted to the CAS for publication). A stripped down version of the monograph becomes an AAS-software series paper - which extends the existing AAS paper on just the `Aggregate` class. A different stripped down version becomes the CAS Education offering.

**Six steps**
1. Publish beta tag
2. Fix aggregate Loss Laboratory (aLL) against beta
3. Sort docs &c
4. Finalize v1.0
5. Polish release materials (press release etc.)
6. Release

## Milestones
1. Publish a Beta tag, which requires
	1. Fixed interface, validated through automated ability to check FCC &c coverage -> requires complete set of programs that **exercise** the API
	2. Docs build with complete auto-code coverage (ch 3 and 4), but no extension to existing Ch 2 and 5 (and see below)
	3. Example exercising-libraries complete - extended use of existing `notes{...}` capability for tags, keywords, narrative purpose, groups. Illustrating features bs stress testing all options. Type of example?
	4. Update cheat sheets (auto and easy)
	5. Update README as GitHub front-door
	6. After tagging 1.0.0b1: Merge refactor branch, start new beta branch
	7. No release publicity

2. Fix aggregate Loss Laboratory (aLL) SPA and website finalized against beta w heroes and example explain-libraries; first publicity opportunity.

3. We *then* finalize supporting materials
	1. **Traditional Docs** (rst) excluding ch 2 and 5, leaving a cleaner documentation build; (semi auto and reasonably easy; they already build w beta tag)
	2. Build CAS **Monograph** (quarto qmd format) (semi auto) (theory / practice=Cookbook)
		1. Base = Ch 2 and 5 of docs -> can you do a rst to qmd port?
		2. Cookbook formalism and consistent examples (recipe objective; decl; build & validate; exhibits; the check ("beat 4")) (covers 2.5 to 2.13 of current docs)
		3. Reproducing/extending published examples folded into same cookbook design/process (each becomes a recipe)
	3. **AAS** draft "white" paper (semi auto) = much shorter, tighter version of Monograph
    4. **Education manifesto** and CAS entry (9/18 deadline) = learning objective wrap around Monograph (separate and non-blocking for v1.0)
    5. Delete ch 2 and 5 from existing docs

4. Finalize v1.0 and "quiet publish" push to GHub

5. Publicity and press release kit for whole launch (agg, API, docs + Monograph/Cookbook, strong draft for monograph,  strong draft for AAS)

6. Release v1.0 with all supporting material

Monograph chapters
1. General theory
2. Class overview
3. DecL
4. Cookbook

Existing docs use nice header format

**Objectives**: Application of the Portfolio class to capital modeling, including VaR, TVaR, and risk visualization and quantification. Covers material on CAS Part 9. \
**Audience**: ERM, capital modeling, risk management actuaries. \
**Prerequisites**: DecL, aggregate distributions, risk measures. \
**See also**: Catastrophe Modeling, Strategy and Portfolio Management, Case Studies. \
**Contents**: 1. Helpful References

### 1. Beta tag `1.0.0b1` — no publicity

- [ ] FEATURES.csv defines the public interface coverage table and provides an automated check: iterate until it is correct.
- [ ] Known punch list
    - [ ] pollaczeck_khinchine audits poisson freq
    - [ ] Weiner-Hopf factorization extension of PK
    - [ ] pedagogy w-h version
    - [ ] ZM and ZT frequency adjustments
- [ ] Extend `tests/test_agg_libraries.py`. Today it only checks each example **parses**.
      It should check the example **builds**, and that the result has the methods it should
      (`valid`, `validation_explanation`, `summary_df`, `stats_df`, `plot`) — and that its
      check passes. This one change is what makes step 1.2 real
- [ ] `cookbook.agg` covers every feature you are freezing
- [ ] `decl-testers.agg` still fails where it is meant to fail
- [ ] Delete `_test_suite.agg` and `_test_suite2.agg`. Then also retire the old snapshot test that reads them (`test_spec_matches_snapshot` and `tests/data/expected_specs.json`) and fix the Testing section of `CLAUDE.md`, which still says they are the main test
- [ ] `uv run pytest -m 'slow or not slow'` green (plain `pytest` skips the slow bivariate suites)
- [ ] Extend `examples.agg` notes with tags, keywords, and purpose; make
      `aggregate_api/examples.py` read them
- [ ] Tune the section A showcase examples — they are still marked draft
- [ ] Docs chapters 3 and 4 build with full code coverage; you run the build, warnings fatal
- [ ] Chapters 2 and 5: frozen, nothing added, **nothing removed yet**
- [ ] Cheat sheets regenerated
- [ ] README updated as the GitHub front door
- [ ] CHANGELOG, LICENSE, version all agree
- [ ] Install the beta into a fresh empty environment, run one showcase example, check the number
- [ ] Tag `1.0.0b1`, merge `REFACTOR`, start the beta branch

### 2. Finalize the API against the beta — first chance to talk about it

- [ ] Pick the one show-off example (see below)
- [ ] Put it in four places, written once: the playground landing page, the README, the
      5-minute intro, and the first line of the announcement
- [ ] Playground opens with it **already run** — nothing to click
- [ ] While writing the cookbook, every interface problem you hit becomes `b2`, `b3`, …
      with a CHANGELOG line. Never a silent fix
- [ ] Declare the interface final

### 3. Supporting material

- [ ] Convert chapter 5 to Quarto — mostly prose and maths, so this converts well
- [ ] Calibrate the cookbook house book (`docs/cookbook/_setup.py`) — **yours, and nine cookbook sections are waiting on it**
- [ ] Write the cookbook recipes. Chapter 2 material gets **rewritten** as recipes, not converted (see below)
- [ ] Claude converts rst to qmd
- [ ] Fold the published-example reproductions in as recipes
- [ ] Monograph builds on its own
- [ ] **Only now**: strip chapters 2 and 5 out of the docs, and check the docs still build
- [ ] AAS paper: shorter, tighter cut of the monograph
- [ ] Check every citation key exists in `uber-library.bib` — never invent one
- [ ] Education manifesto (CAS, 18 September) — does **not** block v1.0

### 4. Quiet publish

- [ ] Fresh-environment install test on the real 1.0.0
- [ ] Tag `1.0.0`, push, publish to PyPI
- [ ] Say nothing yet — a tag cannot be taken back, so it has to be right first

### 5. Publicity kit

- [ ] One-paragraph announcement built round the show-off example
- [ ] LinkedIn post, long and short
- [ ] Links: playground, docs, monograph draft, AAS draft
- [ ] Zenodo DOI refreshed

### 6. Release

- [ ] Announce, with everything live

---

## What I changed in your draft, and why

**Your "illustrating features vs stress testing — type of example?" question answers
itself.** It is both, but in different files, and they should never merge.
`cookbook.agg` and `decl-testers.agg` are for testing — do they cover everything, does
the broken stuff still break. `examples.agg` is for showing — it is what a new user
meets. An example trying to do both does neither well. That split already exists on
disk; keep it.

**`examples.agg` is already the thing you were about to build.** Its own header says it
feeds three places — the default `build`, the intro, and the playground dropdown. The
`note{}` is the description, the A–K letters are the tags, and section A already says
"landing-page heroes". It just is not being tested. That is the one missing wire, and it
is why step 1.3 above is short.

**Small fix to your framing.** You wrote "Programs that explain the API carries
essentially none." That should be **prose**, not programs — the whole distinction is
programs versus words.

**When a recipe's check is awkward to write, that is the interface telling you
something.** Keep a running list while writing the cookbook. It turns "have I got the
interface right?" from a worry into a list you can work through.

**Chapter 5 converts; chapter 2 should not.** Chapter 5 is 9,126 lines with only 52 code
cells — one every 175 lines, so it is mostly prose and maths and pandoc handles it.
Chapter 2 is 4,870 lines with **397** code cells — one every twelve lines. Those cells
all have to be rewritten as recipes anyway, so converting them first means doing the job
twice. Read chapter 2 for the ideas; write the recipes fresh.

**Never delete anything until its replacement works.** Chapters 2 and 5 stay in the docs,
frozen, until the monograph and cookbook actually cover them and the docs still build.
Otherwise "the docs build cleanly" is true of a docs set that quietly lost half its
content. Same rule for `_test_suite*.agg`.

**A beta nobody uses tells you nothing.** You are not announcing the beta, so the only
thing exercising it is you writing the cookbook against it. That is fine — it is the
plan — but it means cookbook writing is partly how you *find* interface problems, not
just how you describe them. Hence the `b2`, `b3` rule in step 2.

**You do not have to be right about everything at once.** A published "still moving"
list makes the freeze much smaller and much easier to be right about, and it is honest.

---

## The show-off example

What to look for: the shortest program whose answer would cost a good actuary a day of
simulation, and which explains itself with nobody standing next to it.

- **`A.ExposureRating`** — my pick to lead. A premium × limit × loss-ratio table with a
  reinsurance cession, six lines. Every pricing actuary already has that exact table on
  a spreadsheet.
- **`B.ThreeDice`** — my pick to follow it. One line, and the validation error is exactly
  zero. It is the only example that lets someone check "exact, not approximate" by hand.
  Weak on its own, excellent as the proof.
- **`F.NetCeded`** — gross, ceded and net in one object. The best "you cannot do this
  elsewhere" example, but harder to explain in three lines.

Not `A.CatXOLTower`: it carries `hints{bs=2;log2=16;normalize=False;}`, and showing grid
settings in the shop window undercuts "it just works".

**One copy fix.** The README opens with "builds **approximations** to compound
distributions". That fights your own line — *exact, not approximate*. The honest version
is exact **compared to simulation**, not exact compared to arithmetic.

---

## The one date problem

Today is 26 July. The CAS manifesto is due 18 September — 54 days.

The monograph blocks two things at once: your v1.0 publicity, and the manifesto that
wraps round it. Its source is about 14,000 lines. That is a book, and it is the long pole
by a distance.

**What I would do:** the only monograph work that blocks v1.0 is the chapter 5 conversion
plus the cookbook recipes. Let the chapter 2 migration carry on after v1.0 is out —
chapter 2 keeps working in the docs until the recipes replace it, which the
never-delete-before-replacing rule requires anyway.

**If instead the whole monograph must be ready first**, v1.0 lands in September at the
earliest and you write the manifesto against a moving target in the weeks it is due. Your
call; the plan works either way.

---

## Who does what

You: the house book calibration, the maths, the show-off pick, and the writing voice.
Me: the test wiring, the coverage table, the chapter 5 conversion script, link and drift
checking, and the boring completeness sweeps.
