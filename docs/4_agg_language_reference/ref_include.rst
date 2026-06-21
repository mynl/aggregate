.. code-block:: lark

    // Grammar for the DecL (Declarative Language) DSL used by aggregate/parser.py.
    //
    // Loaded with:
    //   Lark.open('decl.lark', start='answer', parser='earley',
    //             lexer='dynamic', maybe_placeholders=True)
    //
    // Migrated from SLY (LALR + hand-tuned %prec hacks) in 2026. Earley dissolves
    // the 16 prior shift/reduce conflicts and the dynamic lexer makes keyword/ID
    // disambiguation contextual, so the SLY `ID['keyword'] = TOKEN` remapping
    // trick is replaced by terminal priorities (keywords priority 2, ID priority
    // 1, builtin-dotted names priority 3).
    //
    // Each alternative carries a `-> alias` so the UnderwritingTransformer in
    // parser.py can dispatch on it.
    
    // ======================================================================
    // Top-level
    // ======================================================================
    
    answer: sev_out          -> answer_sev
          | agg_out          -> answer_agg
          | pnl_out          -> answer_pnl
          | port_out         -> answer_port
          | bv_out           -> answer_bv
          | distortion_out   -> answer_distortion
          | expr             -> answer_expr
    
    // ======================================================================
    // Distortion
    // ======================================================================
    
    // A distortion is a kind id followed by a flat list of its natural
    // parameters (``distortion D ph 0.9``, ``distortion D power 0 100 2``). The
    // kind->parameter-name mapping lives on the Distortion subclass
    // (``decl_params``) and is applied by ``Distortion.decl_spec`` -- the grammar
    // and parser hold no per-kind knowledge. The ``minimum``/``mixture``
    // combinators take a list of distortion *references* instead, so they keep
    // their own productions.
    distortion_out: DISTORTION name ID numberl                                            -> distortion_out_params
                  | DISTORTION name ID buildin_dist_list                                  -> distortion_out_combo
                  | DISTORTION name ID buildin_dist_list WEIGHTS "[" numberl "]"          -> distortion_out_combo_wtd
    
    buildin_dist_list: buildin_dist_list BUILTIN_DIST   -> buildin_dist_list_cons
                     | BUILTIN_DIST                      -> buildin_dist_list_one
    
    // ======================================================================
    // Portfolio
    // ======================================================================
    
    port_out: PORT name trailer agg_list
    
    agg_list: agg_list port_item   -> agg_list_cons
            | port_item            -> agg_list_one
    
    // A portfolio unit is an ordinary loss aggregate (``agg``) or a
    // premium-minus-loss aggregate (``pnl``). Both transform to an
    // ("agg", name, spec) tuple, so the rest of the portfolio path is
    // unchanged. ``?port_item`` is inlined (single child) so the
    // transformer sees the underlying agg_out / pnl_out result directly.
    ?port_item: agg_out
              | pnl_out
    
    // ======================================================================
    // Aggregate
    // ======================================================================
    
    agg_out: AGG name exposures layers sev_clause occ_reins freq agg_reins approx_clause trailer  -> agg_out_full
           | AGG name dfreq      layers sev_clause occ_reins         agg_reins approx_clause trailer  -> agg_out_dfreq
           | AGG name TWEEDIE expr expr expr trailer                                    -> agg_out_tweedie
           | AGG name builtin_agg occ_reins agg_reins trailer                           -> agg_out_rename
           | builtin_agg agg_reins trailer                                              -> agg_out_builtin
    
    // Method-of-moments approximation directive. ``approximate KIND`` (KIND in
    // {exact, sgamma, slognorm}) replaces the freq x sev convolution with a single
    // continuous severity fitted to the aggregate's first three moments (shifted
    // gamma / shifted lognormal), carried on a fixed-1 frequency. ``exact`` (or the
    // omitted clause) is the inert default. Validated in the transformer; the kind
    // word is an ordinary ID. See dev/done/plan-approximate.md.
    approx_clause: APPROXIMATE ID   -> approx_set
                 |                  -> approx_none
    
    // ======================================================================
    // Profit-and-loss aggregate (premium minus loss)
    // ----------------------------------------------------------------------
    // ``pnl NAME <premium> prem - <loss-agg-body>`` is a sibling of ``agg``.
    // The premium is a single deterministic amount subtracted ONCE for the
    // book (an aggregate-level affine shift), as opposed to a constant inside
    // sev/dsev/ssev which is per-claim (multiplied by frequency). The body
    // after ``prem -`` is an ordinary aggregate; only the exposure head is
    // specialised so a bare ``lr`` can bind to the stated premium. See
    // dev/done/plan-pnl-premium.md.
    // ======================================================================
    
    pnl_out: PNL name numbers PREMIUM MINUS pnl_exposures layers sev_clause occ_reins freq agg_reins approx_clause trailer  -> pnl_out_full
           | PNL name numbers PREMIUM MINUS dfreq layers sev_clause occ_reins              agg_reins approx_clause trailer  -> pnl_out_dfreq
    
    pnl_exposures: numbers CLAIMS   -> pnl_exp_claims
                 | numbers LOSS     -> pnl_exp_loss
                 | numbers LR       -> pnl_exp_lr
    
    // ======================================================================
    // Bivariate aggregate (copula-coupled, strictly two-axis)
    // ----------------------------------------------------------------------
    // ``bivariate NAME <shared-count> <two component aggs> copula KIND P
    //   [freq]`` couples the two components' per-claim severities by a copula and
    // accumulates them with a shared outer frequency via a 2D FFT. The body is
    // exactly TWO ``agg`` / ``pnl`` declarations (validated in the class). The
    // trailing ``freq`` is optional (defaults to poisson). See dev/plan-mv.md.
    //
    // The three view-pair prefixes ``netceded`` / ``grossceded`` / ``grossnet``
    // take one ordinary ``agg`` carrying occurrence reinsurance and build the joint
    // per-occurrence aggregate of the named pair of {gross, ceded, net}. The
    // keyword names the pair x-then-y, so ``netceded`` -> x=net, y=ceded;
    // ``grossceded`` -> x=gross, y=ceded; ``grossnet`` -> x=gross, y=net (gross
    // leads when present). See dev/plan-mv.md S3.
    //
    // The ``clash`` statement derives the shared event count and the two per-event
    // Bernoulli triggers from interpretable claim counts (na, nb, nc). See S7.3.
    // ======================================================================
    
    bv_out: BIVARIATE name exposures bv_body copula_clause freq trailer  -> bv_out_copula
          | BIVARIATE name exposures bv_body copula_clause trailer        -> bv_out_copula_nofreq
          | BIVARIATE name dfreq bv_body copula_clause trailer            -> bv_out_copula_dfreq
          | BIVARIATE name exposures dbvsev freq trailer                  -> bv_out_discrete
          | BIVARIATE name exposures dbvsev trailer                       -> bv_out_discrete_nofreq
          | BIVARIATE name dfreq dbvsev trailer                           -> bv_out_discrete_dfreq
          | NETCEDED agg_out                                              -> bv_out_netceded
          | GROSSCEDED agg_out                                            -> bv_out_grossceded
          | GROSSNET agg_out                                              -> bv_out_grossnet
          | CLASH name expr expr expr CLAIMS clash_comp clash_comp freq trailer  -> clash_out
          | CLASH name expr expr expr CLAIMS clash_comp clash_comp trailer        -> clash_out_nofreq
    
    // A discrete bivariate severity: the joint per-claim probability matrix given
    // directly on an explicit lattice, the 2-D analogue of ``dsev``. Three surface
    // forms, auto-detected (Earley + dynamic lexer): a dense contingency table
    // ``dbvsev [xs] [ys] [[row] [row] ...]`` (matrix[i][j] = P(X=xs[i], Y=ys[j])),
    // the same with the matrix omitted (uniform over the lattice), or a sparse list
    // of ``[x y p]`` triples. ``doutcomes`` is ``[numbers]`` (never ``[[...]]``), so
    // the dense / sparse forms never compete. See dev/done/plan-bv-discrete.md.
    dbvsev: DBVSEV doutcomes doutcomes dprob_matrix   -> dbvsev_dense
          | DBVSEV doutcomes doutcomes                -> dbvsev_dense_uniform
          | DBVSEV dtriple_list                       -> dbvsev_sparse
    
    dprob_matrix: "[" dmatrix_rows "]"
    dmatrix_rows: dmatrix_rows drow   -> dmatrix_rows_cons
                | drow                -> dmatrix_rows_one
    drow: "[" numberl "]"            -> drow
    
    dtriple_list: "[" dtriples "]"
    dtriples: dtriples dtriple        -> dtriples_cons
            | dtriple                 -> dtriples_one
    dtriple: "[" expr expr expr "]"   -> dtriple
    
    // A clash component is a limit (optional layers) plus a severity clause; the
    // shared event count and the two per-event Bernoulli triggers are derived from
    // the (na, nb, nc) claim counts by the solver (dev/plan-mv.md S7.3, App. B).
    clash_comp: layers sev_clause   -> clash_comp
    
    bv_body: bv_body bv_item   -> bv_body_cons
           | bv_item           -> bv_body_one
    
    // A component is an ordinary loss aggregate (``agg``) or a premium-minus-loss
    // aggregate (``pnl``); both transform to an ("agg", name, spec) tuple.
    ?bv_item: agg_out
            | pnl_out
    
    // The copula clause is optional: omitted (or ``copula independent``) means the
    // independence copula. ``copula KIND`` (no param) is the parameter-free form
    // (independent); ``copula KIND P`` carries the kind's natural parameter.
    copula_clause: COPULA ID numbers   -> copula_one_param
                 | COPULA ID            -> copula_no_param
                 |                      -> copula_none
    
    // ======================================================================
    // Severity output (standalone `sev X ...` definitions)
    // ======================================================================
    
    sev_out: SEV name sev trailer    -> sev_out_sev
           | SEV name dsev trailer   -> sev_out_dsev
    
    // ======================================================================
    // Frequency
    // ======================================================================
    
    freq: freq ZM expr             -> freq_zm
        | freq ZT                  -> freq_zt
        | MIXED ID expr expr       -> freq_mixed_two
        | MIXED ID expr            -> freq_mixed_one
        | FREQ expr expr           -> freq_two
        | FREQ expr                -> freq_one
        | FREQ                     -> freq_zero
    
    // ======================================================================
    // Reinsurance
    // ======================================================================
    
    agg_reins: AGGREGATE NET OF reins_list    -> agg_reins_net
             | AGGREGATE CEDED TO reins_list  -> agg_reins_ceded
             |                                -> agg_reins_none
    
    occ_reins: OCCURRENCE NET OF reins_list   -> occ_reins_net
             | OCCURRENCE CEDED TO reins_list -> occ_reins_ceded
             |                                -> occ_reins_none
    
    reins_list: reins_list AND reins_clause   -> reins_list_cons
              | reins_clause                  -> reins_list_one
              | tower                         -> reins_list_tower
    
    reins_clause: expr XS expr                   -> reins_clause_xs
                | expr SHARE_OF expr XS expr     -> reins_clause_share
                | expr PART_OF expr XS expr      -> reins_clause_part
                | expr OF expr XS expr           -> reins_clause_of
    
    // ======================================================================
    // Severity (continuous: scipy.stats wrappers)
    // ======================================================================
    
    sev_clause: SEV sev          -> sev_clause_sev
              | SSEV sev         -> sev_clause_ssev
              | dsev             -> sev_clause_dsev
              | BUILTIN_SEV      -> sev_clause_builtin
    
    sev: sev "!"                 -> sev_unconditional
       | sev picks               -> sev_picks
       | sev2 weights splice     -> sev_weighted
       | BUILTIN_SEV             -> sev_builtin
    
    sev2: sev1 PLUS numbers      -> sev2_add
        | sev1 MINUS numbers     -> sev2_sub
        | numbers MINUS sev1     -> sev2_rsub
        | sev1                   -> sev2_passthrough
    
    sev1: numbers TIMES sev0     -> sev1_scaled
        | sev0                   -> sev1_passthrough
    
    sev0: ids numbers CV numbers   -> sev0_mean_cv
        | ids numbers numbers      -> sev0_two_params
        | ids numbers              -> sev0_one_param
        | ids xps                  -> sev0_xps
        | ids                      -> sev0_zero_params
    
    xps: XPS doutcomes dprobs
    
    dsev: DSEV doutcomes dprobs   -> dsev_main
        | dsev "!"                -> dsev_unconditional
    
    dfreq: DFREQ doutcomes dprobs
    
    picks: PICKS "[" numberl "]" "[" numberl "]"
    
    doutcomes: "[" numberl "]"                          -> doutcomes_list
             | "[" expr RANGE expr "]"                  -> doutcomes_range
             | "[" expr RANGE expr RANGE expr "]"       -> doutcomes_range_step
    
    dprobs: "[" numberl "]"   -> dprobs_list
          |                   -> dprobs_none
    
    weights: WEIGHTS EQUAL_WEIGHT expr     -> weights_equal
           | WEIGHTS "[" numberl "]"        -> weights_list
           |                                 -> weights_none
    
    splice: SPLICE "[" numberl "]" "[" numberl "]"   -> splice_two
          | SPLICE "[" numberl "]"                   -> splice_one
          |                                           -> splice_none
    
    // ======================================================================
    // Layers
    // ======================================================================
    
    layers: numbers XS numbers   -> layers_xs
          | tower                -> layers_tower
          |                      -> layers_none
    
    tower: TOWER doutcomes
    
    // ======================================================================
    // Trailer: optional `note{...}` annotation + `hints{...}` build settings.
    // ----------------------------------------------------------------------
    // `note{...}` is pure free-text; `hints{...}` carries `key=value;` build
    // settings (parsed in the underwriter, not here). Both are optional and
    // order-free, at most one of each.
    //
    // Spelled out as five token-distinct alternatives rather than two
    // empty-producing nonterminals (`note hints | hints note`): the latter has
    // two parses of the EMPTY trailer (note_none+hints_none vs hints_none+
    // note_none), which Earley flags as ambiguous. Here every alternative
    // matches a distinct token sequence (NOTE HINTS / HINTS NOTE / NOTE / HINTS
    // / nothing) and `note{` vs `hints{` have disjoint prefixes, so the parse
    // is unambiguous.
    // ======================================================================
    
    trailer: NOTE HINTS   -> trailer_nh
           | HINTS NOTE    -> trailer_hn
           | NOTE          -> trailer_note
           | HINTS         -> trailer_hints
           |               -> trailer_none
    
    // ======================================================================
    // Exposures
    // ======================================================================
    
    exposures: numbers CLAIMS                       -> exposures_claims
             | numbers LOSS                         -> exposures_loss
             | numbers PREMIUM AT numbers LR        -> exposures_premium_lr
             | numbers EXPOSURE AT numbers RATE     -> exposures_exposure_rate
    
    // ======================================================================
    // IDs (severity-name lists or singletons)
    // ======================================================================
    
    ids: "[" idl "]"   -> ids_list
       | ID            -> ids_single
    
    idl: idl ID        -> idl_cons
       | ID            -> idl_one
    
    // ======================================================================
    // Builtin aggregate operations (scale, translate, lookup)
    // ======================================================================
    
    builtin_agg: expr INHOMOG_MULTIPLY builtin_agg   -> builtin_agg_inhomog
               | expr TIMES builtin_agg              -> builtin_agg_homog
               | builtin_agg PLUS expr               -> builtin_agg_plus
               | builtin_agg MINUS expr              -> builtin_agg_minus
               | BUILTIN_AGG                         -> builtin_agg_lookup
    
    // ======================================================================
    // Name (single identifier wrapper)
    // ======================================================================
    
    name: ID
    
    // ======================================================================
    // Numbers (vectors and scalars)
    // ======================================================================
    
    numbers: "[" numberl "]"                            -> numbers_list
           | "[" expr RANGE expr "]"                    -> numbers_range
           | "[" expr RANGE expr RANGE expr "]"         -> numbers_range_step
           | expr                                       -> numbers_scalar
    
    numberl: numberl expr   -> numberl_cons
           | expr           -> numberl_one
    
    // ======================================================================
    // Expression atoms (the DecL math sub-language)
    // ----------------------------------------------------------------------
    // Precedence (tightest first):  ()  >  EXP  >  EXPONENT (** / ^)  >  /
    // EXPONENT is right-associative (2**3**4 == 2**(3**4));
    // DIVIDE is left-associative (1/2/3 == (1/2)/3).
    // `?term` and `?factor` are inlined when they have a single child, so
    // the transformer only sees the aliased nodes (atom_divide, atom_exp,
    // atom_exponent, atom_parens, atom_number).
    // ======================================================================
    
    ?expr: term
    
    ?term: term DIVIDE factor       -> atom_divide
         | factor
    
    ?factor: EXP factor             -> atom_exp
           | atom EXPONENT factor   -> atom_exponent
           | atom
    
    atom: NUMBER                    -> atom_number
        | "(" expr ")"              -> atom_parens
    
    // ======================================================================
    // Terminals
    // ======================================================================
    
    // Keywords — priority 2 so they outrank the catch-all ID terminal.
    //
    // Each keyword carries a negative lookahead `(?![a-zA-Z0-9._:~\-])` matching
    // the ID-continuation character class. Without it, Lark's dynamic lexer would
    // happily peel `agg` off the front of a typo like `aggx`, letting the parse
    // progress into the wrong rule and surface the error at the wrong column.
    // The lookahead forces keywords to match only on word boundaries — same trick
    // Python's tokenizer uses to distinguish `def` from `define`.
    OCCURRENCE.2: /occurrence(?![a-zA-Z0-9._:~\-])/
    AGGREGATE.2:  /aggregate(?![a-zA-Z0-9._:~\-])/
    APPROXIMATE.2: /(?:approximate|approx)(?![a-zA-Z0-9._:~\-])/
    BIVARIATE.2: /(?:bivariate|bv)(?![a-zA-Z0-9._:~\-])/
    NETCEDED.2:   /netceded(?![a-zA-Z0-9._:~\-])/
    GROSSCEDED.2: /grossceded(?![a-zA-Z0-9._:~\-])/
    GROSSNET.2:   /grossnet(?![a-zA-Z0-9._:~\-])/
    CLASH.2:      /clash(?![a-zA-Z0-9._:~\-])/
    COPULA.2:     /copula(?![a-zA-Z0-9._:~\-])/
    EXPOSURE.2:   /exposure(?![a-zA-Z0-9._:~\-])/
    TWEEDIE.2:    /tweedie(?![a-zA-Z0-9._:~\-])/
    PREMIUM.2:    /(?:premium|prem)(?![a-zA-Z0-9._:~\-])/
    TOWER.2:      /tower(?![a-zA-Z0-9._:~\-])/
    MIXED.2:      /mixed(?![a-zA-Z0-9._:~\-])/
    PICKS.2:      /picks(?![a-zA-Z0-9._:~\-])/
    CLAIMS.2:     /(?:claims|claim)(?![a-zA-Z0-9._:~\-])/
    SPLICE.2:     /splice(?![a-zA-Z0-9._:~\-])/
    CEDED.2:      /ceded(?![a-zA-Z0-9._:~\-])/
    DBVSEV.2:     /dbvsev(?![a-zA-Z0-9._:~\-])/
    DFREQ.2:      /dfreq(?![a-zA-Z0-9._:~\-])/
    DSEV.2:       /dsev(?![a-zA-Z0-9._:~\-])/
    SSEV.2:       /ssev(?![a-zA-Z0-9._:~\-])/
    LOSS.2:       /loss(?![a-zA-Z0-9._:~\-])/
    PNL.2:        /pnl(?![a-zA-Z0-9._:~\-])/
    PORT.2:       /port(?![a-zA-Z0-9._:~\-])/
    RATE.2:       /rate(?![a-zA-Z0-9._:~\-])/
    NET.2:        /net(?![a-zA-Z0-9._:~\-])/
    SEV.2:        /sev(?![a-zA-Z0-9._:~\-])/
    AGG.2:        /agg(?![a-zA-Z0-9._:~\-])/
    XPS.2:        /xps(?![a-zA-Z0-9._:~\-])/
    WEIGHTS.2:    /wts(?![a-zA-Z0-9._:~\-])/
    AND.2:        /and(?![a-zA-Z0-9._:~\-])/
    EXP.2:        /exp(?![a-zA-Z0-9._:~\-])/
    AT.2:         /at(?![a-zA-Z0-9._:~\-])/
    CV.2:         /cv(?![a-zA-Z0-9._:~\-])/
    LR.2:         /lr(?![a-zA-Z0-9._:~\-])/
    XS.2:         /xs(?![a-zA-Z0-9._:~\-])/
    OF.2:         /of(?![a-zA-Z0-9._:~\-])/
    TO.2:         /to(?![a-zA-Z0-9._:~\-])/
    PART_OF.2:    /po(?![a-zA-Z0-9._:~\-])/
    SHARE_OF.2:   /so(?![a-zA-Z0-9._:~\-])/
    ZM.2:         /zm(?![a-zA-Z0-9._:~\-])/
    ZT.2:         /zt(?![a-zA-Z0-9._:~\-])/
    DISTORTION.2: /(?:distortion|dist)(?![a-zA-Z0-9._:~\-])/
    
    FREQ.2: /(?:binomial|pascal|poisson|bernoulli|geometric|fixed|neymanA|neymana|neyman|logarithmic|negbin)(?![a-zA-Z0-9._:~\-])/
    
    // agg.X / sev.X / dist(ortion).X / note{...} — priority 3 outranks the
    // AGG / SEV / DISTORTION / ID alternatives that share their prefix.
    BUILTIN_AGG.3:  /agg\.[a-zA-Z][a-zA-Z0-9._:~\-]*/
    BUILTIN_SEV.3:  /sev\.[a-zA-Z][a-zA-Z0-9._:~\-]*/
    BUILTIN_DIST.3: /(?:distortion|dist)\.[a-zA-Z][a-zA-Z0-9._:~\-]*/
    NOTE.3:         /note\{[^}]*\}/
    HINTS.3:        /hints\{[^}]*\}/
    
    // NUMBER absorbs an optional leading minus so `-3` is one token rather than
    // MINUS NUMBER. Priority 2 keeps it ahead of the standalone MINUS terminal.
    // Each digit run is `\d(?:_?\d)*` so Python-style `_` group separators are
    // accepted (`10_000_000`, `1_000.5`, `1_0e3`) while leading / trailing /
    // doubled underscores (`_1`, `1_`, `1__0`) are rejected by the lexer, exactly
    // as Python's float()/int() do. `float('10_000_000')` already strips them, so
    // the transformer needs no change.
    NUMBER.2: /-?(\d(?:_?\d)*\.?(?:\d(?:_?\d)*)?|\.\d(?:_?\d)*)([eE][+\-]?\d(?:_?\d)*)?%?|-?inf/
    
    // ID is the catch-all identifier — priority 1 (default).
    // Two negative lookaheads at the start of the match make the grammar
    // unambiguous:
    //   1. (?!agg\.|sev\.|dist(ortion)?\.) — anything starting with agg.,
    //                                sev., dist. or distortion. is
    //                                BUILTIN_AGG / BUILTIN_SEV / BUILTIN_DIST
    //                                territory.
    //   2. (?!keyword(?![namechar])) — reject a keyword standing alone (i.e.,
    //                                followed by a non-name character or end
    //                                of input). Names containing a keyword as
    //                                a prefix (e.g., `premium_account`,
    //                                `aggressive`) still lex as ID because
    //                                the inner lookahead fails when the next
    //                                character is itself a name character.
    // Without these lookaheads, Earley + the dynamic lexer would explore
    // both the keyword and ID interpretations for inputs like `dsev` or
    // `sev.One`, leaving the grammar ambiguous and relying on tie-breaker
    // heuristics to land on the intended parse.
    ID: /(?!agg\.|sev\.|dist\.|distortion\.)(?!(?:agg|aggregate|and|approximate|approx|at|bernoulli|binomial|bivariate|bv|ceded|claim|claims|clash|copula|cv|dbvsev|dfreq|dist|distortion|dsev|exp|exposure|fixed|geometric|grossceded|grossnet|logarithmic|loss|lr|mixed|negbin|net|netceded|neyman|neymana|neymanA|occurrence|of|pascal|picks|pnl|po|poisson|port|prem|premium|rate|sev|so|splice|ssev|to|tower|tweedie|wts|xps|xs|zm|zt)(?![a-zA-Z0-9._:~\-]))[a-zA-Z][\._:~a-zA-Z0-9\-]*/
    
    EXPONENT:         "**" | "^"
    PLUS:             "+"
    MINUS:            "-"
    TIMES:            "*"
    DIVIDE:           "/"
    INHOMOG_MULTIPLY: "@"
    EQUAL_WEIGHT:     "="
    RANGE:            ":"
    
    // Whitespace + noise: space, tab, comma, pipe. The backslash was dropped when
    // `\`-continuation was removed (blank-line / `;` statement separation), so a
    // stray backslash now raises a clear lexer error instead of vanishing.
    %ignore /[ \t,|]+/
