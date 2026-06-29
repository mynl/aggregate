> **SUPERSEDED 2026-06-29** — moved to `dev/done`; superseded by `dev/plan-bivariate-legs.md` (see its consolidation note). Executed (Phase 1).

# Plan — multi-stage Gross/Ceded/Net `PnL` (occurrence and/or aggregate reins)

> **Status: DRAFT — not executed.** Generalizes the single-stage (aggregate-only)
> GCN view shipped in `1.0.0a106` (`dev/done/plan-pnl.md` §3.4). Goal: a `PnL`
> over a reinsurance-bearing risky leg presents the **whole cession chain** —
> gross, ceded/net per occurrence stage, ceded/net per aggregate stage — with a
> premium per stage, and **omits stages that aren't configured**.

---

## Why (the a106 restriction)

a106 keyed the GCN legs off `agg.agg_density_gross/ceded/net`, which the reins
machinery populates **only for aggregate covers** (`apply_agg_reins`). So GCN
today requires aggregate reinsurance and silently excludes occurrence-only
treaties. It also uses `p_agg_subject` (the *post-occ* input to the agg cover) as
"gross", which **understates true gross** on a program carrying *both* occ and
agg covers.

**The fix is to source from `reins_density_df`,** which is defined for
occurrence, aggregate, and combined covers and exposes every aggregate-level
view:

| view | `reins_density_df` column |
|---|---|
| true gross aggregate `L_g` | `p_agg_gross` |
| ceded occurrence (recovery) `R_o` | `p_agg_ceded_occ` |
| net of occurrence `L_no` | `p_agg_net_occ` |
| ceded aggregate (recovery) `R_a` | `p_agg_ceded` |
| net of aggregate (final) `L_na` | `p_agg_net` |

Missing-stage columns default to the no-cession values, so a single source
handles all four configurations (none / occ / agg / both).

---

## The cession chain (all means add, comonotone → stays 1-D)

```
gross            L_g
  − ceded occ    R_o            net occ   L_no = L_g − R_o
  − ceded agg    R_a            net agg   L_na = L_no − R_a   (= what comes out)
```

Premiums (signed; received `+`, paid `−`): gross `Pg`, ceded-occ `Po`,
ceded-agg `Pa`. Net considerations accumulate down the chain:
`Pn_o = Pg − Po`, `Pn_a = Pg − Po − Pa`.

**Exhibit (`gcn_df`)** — one row per *present* stage, the signed-additive
convention of `summary_df` (loss subtracts, recovery adds):

| leg | Consideration | Obligation | Margin |
|---|---|---|---|
| Gross | `+Pg` | `−E[L_g]` | `Pg − E[L_g]` |
| Ceded occ | `−Po` | `+E[R_o]` | `E[R_o] − Po` |
| Net occ | `Pg−Po` | `−E[L_no]` | `… ` |
| Ceded agg | `−Pa` | `+E[R_a]` | `E[R_a] − Pa` |
| Net agg | `Pg−Po−Pa` | `−E[L_na]` | `…` |

Additive **both ways**: each Net row = prior Net (or Gross) + the Ceded stage;
Margin = Consideration + Obligation. **Net agg is the headline** — it drives the
net `density_df`, moments, `evaluate`, and `plot`. Omit rows for absent stages
(occ-only → Gross/Ceded occ/Net occ; agg-only → Gross/Ceded agg/Net agg = the
a106 case).

---

## API

Redesign `make_pnl` (and `PnL.__init__`):

```python
agg_re.make_pnl(consideration=P)                       # net-only (unchanged)
agg_re.make_pnl(gross=Pg, ceded_occ=Po, ceded_agg=Pa)  # full chain
agg_re.make_pnl(gross=Pg, ceded_agg=Pa)                # agg-only  (a106)
agg_re.make_pnl(gross=Pg, ceded_occ=Po)                # occ-only  (new)
```

- Validate the premiums supplied against the **stages actually present** on the
  leg (`occ_reins` / `agg_reins`): a `ceded_occ=` with no occurrence cover is an
  error, etc.
- Source loss views from `reins_density_df` (drop the `agg_density_*` path;
  fixes the true-gross gap).
- `net=` override and the constant-vs-single-leg behavior carry over from a106.

**Open questions:** kwarg names (`ceded_occ`/`ceded_agg` vs `occ`/`agg` vs a dict
`ceded={'occ':…, 'agg':…}`); whether a single `ceded=` on a one-stage program is
sugar; how a `net=` override composes with a multi-stage chain.

## Tests / corpus

Reins programs with occ-only, agg-only, and both; assert the present rows, the
double additivity, true-gross under occ+agg, and that Net agg drives the moments.
Plot overlays the present legs. Python-API only (no DecL surface).
