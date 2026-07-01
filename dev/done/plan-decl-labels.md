# Plan — [DecL-Labels]: human labels, quoted names, and expense grouping

Status: **DONE** (`1.0.0a124`, 2026-07-01). Landed before
`dev/plan-pnl-engine-source.md` (still DRAFT) so its program sweep is written once
against final syntax. Pure-presentation: no computed value changed.

**Delivered:** `STRING` terminal + reserved `as`; `display_label` /
`display_name` on `agg` / `pnl` / `sev` / `port` (repr + exhibit titles as
`label (name)`); premium label → consideration leg key; two-level expense grammar
(`and` combines, juxtaposition separates, per-group `as` label, basis-derived
defaults, single-group back-compat `expense` leg); reins-cession label →
`margin_df` column; full unparser round-trip; `tests/test_decl_labels.py` (21
cases) + `decl-testers.agg` §Y. **Deferred:** per-component labels inside a
*mixture* severity (invasive severity mini-language change for marginal value —
object-level `sev` label delivered); tracked in `dev/TODO.md`.

---

Status (original): **DRAFT** (2026-07-01). Pure-presentation companion to
`dev/plan-pnl-engine-source.md`. **Recommended to land first** (see that plan's
ordering note): additive, low-risk, and independent of the structural refactor,
so the engine-source program sweep is written once against final syntax.

Follows the a123 `[PnL-Exhibits]` work (`dev/done/plan-pnl-exhibits.md`). No
change to any computed value — this is labels landing in dict keys and column
headers. DecL compute is untouched.

## Motivating idea — identity vs. presentation

A DecL object has two separable naming roles, and today it only has the first:

* **handle** — the bareword `name` (`agg GrossBook`). It is the *identity*:
  the knowledge-base key and the target of `agg.GrossBook` / portfolio-unit
  references. Must stay ID-shaped (dotted references are ID-shaped; a spaced name
  is unreferenceable).
* **display label** — an optional human string (`as "Gross Book P&L"`), spaces
  allowed, used for exhibit rows / column headers / repr titles.

This is **not** a synonym for one concept (which the house rule forbids) — it is
the same identity-vs-presentation split as the library's `_description` /
`_explanation` convention. So: keep `name: ID` as the handle, add `as "…"` as the
label. Do **not** quote the handle.

## Decisions (from the design dialogue)

1. **`STRING` terminal**, delimited: `/"[^"\n]*"/` — no embedded newlines (an
   unbalanced quote fails on its own line, not by swallowing the program) and no
   escape handling in v1 (add `\"` only if a label ever needs a literal quote).
   Quotes are a fresh lexical class (DecL is quote-free today), so they cannot
   collide with keywords / `ID` / `numbers` — the safe kind of addition.
2. **`as` becomes a reserved word.** It is *not* in the `ID` exclusion list today;
   add it, and grep the suite / knowledge base for any unit literally named `as`.
3. **`as (ID | STRING)`** everywhere a label is allowed — a bareword label skips
   the quote tax (`as lae`), quotes carry spaces (`as "Loss Adjustment Expense"`).
4. **Expense grouping.** `and` **combines** terms into one reported item;
   **juxtaposition** (no `and`) makes **separate** items. This is consistent with
   the rest of DecL: layers inside one `occurrence net of` joined by `and`
   consolidate to a single net (you cannot get net-by-layer out), and the
   "separate" analog is juxtaposed *clauses* — exactly the whitespace case.
   Backward-compatible: today's `and`-joined expenses already sum to one leg.
5. **Loss labeling defers to naming the agg.** Under `[PnL-Engine-Source]` the
   P&L's loss *is* an `agg` with its own `name` + `as` label, so there is **no**
   floating loss `as` clause (which would have been ambiguous against a reins
   clause's own `as`). One less ambiguity to fight.

## Where each label lands (the real per-attachment work)

The grammar rule is uniform; the labor is that each label must *land somewhere*
and flow to the right exhibit.

| Attachment | Grammar | Lands as | Effort |
|---|---|---|---|
| **object** (`agg`/`pnl`/`sev`/`port`/`bv`) | `… name (AS label)? …` via shared `name` site | a new `display_label` attribute; threads into `repr` / exhibit titles / index names | small (one attribute + a few call sites) |
| **premium** | `pnl_premium (AS label)?` | the consideration leg's dict key | free (labels are data on `PnL`) |
| **expense group** | per group (see below) | the obligation leg's dict key | free |
| **sev component** | `sev … (AS label)?` | the severity component name (esp. mixtures) | small (sev spec carries a name) |
| **reins clause** | `occ_reins / agg_reins (AS label)?` | the `margin_df` perspective **column** label + the attached analysis `.name` | light plumbing (column relabel in the tower/analysis) |

## Grammar sketch

```
// shared
STRING: /"[^"\n]*"/
label:  ID | STRING
AS.2:   /as(?![a-zA-Z0-9._:~\-])/          // new reserved word

// object display label — attach at each `name` site (or fold into `name_label`)
name_label: name (AS label)?

// expenses: two-level list (group of and-joined terms = one item)
expense_clause: expense_group+                         -> expense_some
              |                                          -> expense_none
expense_group:  expense_group AND expense_term (AS label)?  -> expense_group_cons
              | expense_term (AS label)?                    -> expense_group_one
expense_term:   numbers PREMIUM EXPENSES
              | numbers LOSS    EXPENSES
              | numbers FIXED   EXPENSES
```

The `AS label` on a group attaches to the **group**, not each term (a combined
group gets one name); default name from the basis (`loss expense` / `fixed
expense` / `premium expense`) or a generic `expense` for a mixed group.

**Ambiguity check:** each `expense_term` is anchored by the `expense[s]` keyword,
so `and` vs. juxtaposition fully disambiguates (an `and` continues the group; a
bare `numbers…` starts a new group; anything else falls to the trailer). Standard
Earley, low risk. The `AS STRING` is delimited, so it never collides with the
trailer.

## Payload changes (Python)

* `_resolve_expense_split` → return a **list of groups** `[(name, scalar,
  loss_rate), …]` instead of one blended pair; `make_pnl` builds **one obligation
  leg per group** (named). Labels-are-data means `summary_df` / `density_df`
  surface them for free.
* A `display_label` (or `label`) attribute on `Aggregate` / `Portfolio` /
  `Severity` / `PnL`, defaulting to `None` → repr and exhibit titles prefer it
  over `name` when present. Vet the name against the existing surface (no
  collision with `name`, `note`, `_description`).
* Severity component names thread through the sev spec (mixtures already carry
  weights; add an optional per-component label).
* Reins-clause label → the `PnLTower` / analysis column relabel for `margin_df`.

## Phases

1. **[Labels-Lexicon]** — `STRING` terminal, reserve `as`, `label: ID | STRING`,
   the `name_label` site. No behavior yet; snapshot re-baseline for the new token.
2. **[Labels-Objects]** — `display_label` attribute + repr / exhibit-title
   threading for `agg` / `pnl` / `sev` / `port`.
3. **[Labels-Parts]** — premium + sev-component labels (free-ish: dict keys / spec
   names).
4. **[Expense-Grouping]** — the two-level expense grammar + per-group leg build +
   per-group `as` label.
5. **[Labels-Reins]** — reins-clause `as` → `margin_df` column + analysis `.name`.

## Testing / housekeeping

* New `tests/test_decl_labels.py` (quoted names, `as` on each attachment, expense
  grouping combine-vs-separate, default names). Append the DecL programs to
  `src/aggregate/agg/test_decl.agg` under the matching section (house rule).
* Re-baseline the SLY snapshot (`tests/data/expected_specs.json`) for the new
  tokens / spec shapes.
* Version bump (`1.0.0a*`), `CHANGELOG.md` `[DecL-Labels]` section, `dev/TODO.md`
  entry, `dev/FEATURES.csv` if the public surface changes (the `display_label`
  attribute; per-leg expense rows are data, not new members).
* Docs: a short DecL-reference note on `as` + quoting + expense grouping (author
  rebuilds; keep `.rst`/`.qmd` refs in lockstep, do not build in the loop).

## Risks

* **Low overall.** The one real spot is the two-level expense list — verified
  unambiguous by the `expense` keyword anchor, but wants explicit combine /
  separate / mixed-group tests.
* Reserving `as` is a (tiny) breaking change to the reserved set — grep first.
* Reins-clause label plumbing touches the tower/analysis column naming — coordinate
  with the `margin_df` column vocabulary noted in `[Reporting-Guidelines]`.
