# Plan — DecL unary minus on a severity (`ssev -lognorm …`)

> **Status: DRAFT — not executed.** Not PnL-specific. Goal: let DecL accept a
> **bare unary minus** in front of a severity, `ssev -lognorm 10 cv 0.5`, as
> sugar for the working `ssev 0 - lognorm 10 cv 0.5` (reflect the severity).

---

## Current behavior

The severity grammar (`decl.lark`) handles a *binary* minus:

```
sev2: sev1 PLUS numbers      -> sev2_add      // dist + c
    | sev1 MINUS numbers     -> sev2_sub      // dist − c
    | numbers MINUS sev1     -> sev2_rsub     // c − dist  (reflect+shift)
    | sev1                   -> sev2_passthrough
```

So `0 - lognorm …` parses as `sev2_rsub` (numbers `0` MINUS sev1) and reflects.
The `NUMBER` terminal absorbs a leading minus **only before a digit / `inf`**
(`-?(\d…|\.\d…|inf)`), so `-lognorm` lexes as a standalone `MINUS` token followed
by `ID(lognorm)` — i.e. there is a real `MINUS` to hang a rule on.

## Feasibility — **feasible, low-risk** (Lark/Earley, not SLY)

The SLY unary-minus `%prec` pain does **not** apply: the parser is Earley with a
dynamic lexer (migrated 2026), which dissolves the shift/reduce conflicts that
made unary minus fiddly under LALR. The work is a one-line grammar alternative
plus a transformer method.

### Semantics (author-confirmed): standard math precedence — unary minus binds *tighter* than the shift

`-X` reflects the distribution; the additive shift `+/− c` then wraps it, exactly
as in ordinary algebra. Unary minus does **not** swallow a trailing shift:

| input | parses as | meaning |
|---|---|---|
| `-lognorm 100 cv 2` | `-(lognorm)` | reflect, mean `-E[X]` |
| `-lognorm 2 + 5` | `(-lognorm 2) + 5` | `5 - X`, mean `5 - E[X]` |
| `-lognorm 2 - 5` | `(-lognorm 2) - 5` | `-X - 5`, mean `-E[X] - 5` |
| `-lognorm 100 cv 2 + 10` | `(-lognorm) + 10` | `10 - lognorm` |

**Worked example — `-lognorm 2 + 5`** (the canonical case). Let `X ~ lognorm 2`.
Unary minus binds *first* (`-X`), then the `+ 5` shift:

```
-lognorm 2 + 5  ==  (-lognorm 2) + 5  ==  (-X) + 5  ==  5 - X
```

so its mean is `5 - E[X]` — the same RV as the existing `5 - lognorm 2`
(`sev2_rsub`). It is **not** `-(X + 5) = -X - 5` (the rejected "negate the whole
expression" reading — a brain-twister we explicitly avoid). The identity
`-(X + c) = -X - c` still holds, and you write `-X - c` directly as
`-lognorm 2 - c`.

So the grammar negates at the **`sev1`** level (so the negated term can be the
left operand of the `sev2` shift), tight-binding:

```
sev1: numbers TIMES sev0     -> sev1_scaled    // existing
    | MINUS sev1             -> sev1_negate    // NEW: −dist (binds tight)
    | sev0                   -> sev1_passthrough
```

`sev1_negate` reflects its `sev1` child (reuse the `sev2_rsub` reflection with an
implicit `0`; a pure reflection, no shift). `-lognorm + 5` is then
`sev2_add(sev1 = -lognorm, numbers = 5)` = `(-X) + 5` = `5 - X`. (`-3 * lognorm`
does **not** use this rule — `-3` lexes as a single `NUMBER`, so it is the
existing negative-scale `sev1_scaled` path.)

### The one real risk: Earley ambiguity — why `MINUS sev1` stays unique

`MINUS sev1` must not make any input parse two ways:
- `-lognorm + 5` → `sev2_add(sev1 = MINUS lognorm, 5)` only — the `+ 5` attaches
  to the negated `sev1`; no competing parse (there is no `MINUS sev2`).
- `100 - lognorm` → `numbers MINUS sev1` (`sev2_rsub`); the leading `numbers`
  blocks the bare-`MINUS` reading.
- `lognorm - 5` → `sev1 MINUS numbers` (`sev2_sub`) — unaffected.
- bare `lognorm` → `sev1_passthrough` — unaffected.

Add the production, build the grammar, and **check for Lark ambiguity warnings**
on the corpus before relying on it.

## Scope decisions

- **`sev -X` is an error (author-confirmed).** A reflected severity is inherently
  signed; the clamped `sev` clause would silently delete the negative half, which
  is nonsense. So `sev_clause_sev` (the `SEV` path) must **reject a signed /
  reflected severity** with a clear message, e.g. *"a reflected (signed)
  severity needs `ssev`, not `sev`"*. The negate form is only valid under
  `ssev`. (Apply the same guard to the existing `sev 0 - lognorm` if it currently
  clamps silently — same nonsense, same fix.)
- Compose with scale: `ssev -3 * lognorm …` → `-(3·lognorm)` (the negate wraps
  the scaled `sev1` inside its `sev2`). Confirm it falls out.

## Tests / corpus

`ssev -lognorm 10 cv 0.5` parity with `ssev 0 - lognorm 10 cv 0.5`; `ssev -3 *
lognorm`; round-trip through the unparser (`decl_writer` must emit the canonical
form — likely re-emit as `0 - …` or as `-…`, pick one and keep idempotent);
ambiguity-free parse. Mirror in `decl-testers.agg`. Update `ref_include.rst`
grammar (pending manual docs rebuild).
