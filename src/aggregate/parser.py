"""
Aggregate Lexer and Parser.
===========================

Implements the DecL (Declarative Language) DSL used to describe aggregate
distributions. The grammar lives in ``aggregate/decl.lark``; this module
provides the ``UnderwritingLexer`` and ``UnderwritingParser`` wrappers that
``aggregate.underwriter`` consumes.

The parser uses `Lark <https://lark-parser.readthedocs.io/>`_ with an Earley
backend and a dynamic, context-sensitive lexer. Earley dissolves the
shift/reduce conflicts that the previous SLY (LALR) implementation needed to
hand-tune with ``%prec`` hacks, and the dynamic lexer disambiguates keywords
from identifiers based on grammar context rather than the SLY
``ID['keyword'] = TOKEN`` remapping trick.

REPL debugging session::

    from aggregate.parser import UnderwritingLexer, UnderwritingParser

    lexer = UnderwritingLexer()
    parser = UnderwritingParser(lambda x: x, debug=True)

    while True:
        try:
            text = input(">> ")
            if not text:
                continue
        except (EOFError, KeyboardInterrupt):
            break
        try:
            tokens = list(lexer.tokenize(text))
            print("Tokens:")
            for tok in tokens:
                print(f"  {tok.type:<10} {tok.value!r}")
            print("Parsed:", parser.parse(text))
        except Exception as e:
            print("Error:", e)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
from lark import Lark, Transformer
from lark.exceptions import (UnexpectedCharacters, UnexpectedInput,
                             UnexpectedToken, VisitError)

from .parser_errors import format_error

logger = logging.getLogger(__name__)

__all__ = ['UnderwritingLexer', 'UnderwritingParser', 'grammar']

GRAMMAR_FILE = Path(__file__).parent / "decl.lark"


# ======================================================================
# Lexer
# ======================================================================


class _TokenizedText:
    """An iterable of Lark tokens that also remembers the original source text.

    ``UnderwritingParser.parse`` accepts either a raw string or one of these
    objects, preserving the historic ``parser.parse(lexer.tokenize(line))``
    call shape from ``aggregate.underwriter``.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __iter__(self) -> Iterator[SimpleNamespace]:
        for tok in _PARSER.lex(self.text):
            yield SimpleNamespace(
                type=tok.type, value=str(tok), index=tok.start_pos or 0
            )


class UnderwritingLexer:
    """DecL lexer. Thin wrapper around Lark's tokenizer plus a regex
    preprocessor that splits multi-line programs and strips comments."""

    @staticmethod
    def preprocess(program: str) -> list[str]:
        """Split a multi-line DecL program into individual statements.

        Two statements are separated either by a **blank line** (the markdown
        paragraph model — a line that is empty or whitespace-only) or by a
        **semicolon at end of line** (the Python model, so dense
        one-statement-per-line lists stay legal). Every other newline is just
        whitespace, so a single statement may be laid out across as many
        physical lines, with whatever indentation, as the author likes. The
        former ``\\`` line-continuation has been removed; a stray backslash is
        now a lexer error (it was dropped from ``decl.lark``'s ``%ignore``).

        Comments are **transparent**: they never separate statements. A
        full-line comment between the clause-lines of one statement (e.g. a
        commented-out reinsurance clause) simply vanishes; the lines around it
        stay in the same statement. The corollary is that a comment cannot
        separate two statements — use a blank line or a ``;`` for that.

        The preprocessor performs six steps:

        1. Full-line comments (optional indent, then ``#`` / ``//``) are removed
           **entirely, including their newline**, so they leave no blank-line
           ghost and a comment inside a multi-line statement folds away. Done
           first so a stray bracket in a comment can never unbalance step 3.
        2. Trailing (inline) comments are stripped to end of line, keeping the
           line's own newline.
        3. Newlines inside ``[ ]`` (e.g., from formatted numpy arrays) are
           collapsed to spaces, so a vector never reads as a paragraph break.
        4. A ``;`` at end of line is turned into a blank-line break. Only a
           line-final ``;`` fires, so the ``;`` inside ``hints{key=value;}`` /
           ``note{...}`` (which always end a line with ``}``) is untouched.
        5. The text is split into paragraphs on runs of blank lines.
        6. Each paragraph is flattened: its newlines, indentation, and repeated
           spaces collapse to single spaces. Empty paragraphs are dropped.

        Parameters
        ----------
        program : str
            Raw multi-line DecL source.

        Returns
        -------
        list[str]
            Non-empty, whitespace-normalised DecL statements ready for parsing.
        """
        # 1. Remove full-line comments ENTIRELY (line + its newline), so a
        # comment is transparent: it never separates statements and never
        # masquerades as a blank line. Done before the bracket step so a stray
        # bracket in a comment can't unbalance it.
        program = re.sub(r"(?m)^[ \t]*(?://|#)[^\n]*\n?", "", program)

        # 2. Strip trailing (inline) comments to end of line, keeping the newline.
        program = re.sub(r"(//|#)[^\n]*", "", program)

        # 3. Collapse newlines inside [...] (which can appear when a vector is
        # formatted with f'{np.linspace(...)}'). The flat split below assumes
        # brackets do not nest -- true for every form except the dbvsev dense /
        # sparse matrices (``[[ ... ]]``). When nesting is present, fall back to a
        # depth-aware scan that only turns interior newlines into spaces and
        # leaves the bracket structure untouched. The non-nested path is kept
        # byte-for-byte so the captured spec snapshot (keyed on this text) is
        # unaffected.
        depth = max_depth = 0
        for ch in program:
            if ch == "[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == "]":
                depth = max(0, depth - 1)
        if max_depth <= 1:
            out_in = re.split(r"\[|\]", program)
            assert len(out_in) % 2  # must be odd
            odd = [t.replace("\n", " ") for t in out_in[1::2]]
            even = out_in[0::2]
            program = " ".join(
                [even[0]] + [f"[{o}] {e}" for o, e in zip(odd, even[1:])])
        else:
            chars, depth = [], 0
            for ch in program:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth = max(0, depth - 1)
                elif ch == "\n" and depth > 0:
                    ch = " "
                chars.append(ch)
            program = "".join(chars)

        # 4. A line-final ``;`` terminates a statement -> turn it into a blank
        # line. ``;`` inside hints{}/note{} is never line-final (those end in
        # ``}``), so it is left alone.
        program = re.sub(r";[ \t]*(\r?\n|$)", "\n\n", program)

        # 5 + 6. Split on blank-line runs and flatten each paragraph: a newline
        # plus the whitespace around it (an indented continuation line) folds to
        # a single space, but existing intra-line spacing is preserved -- the
        # lexer ignores it, and keeping it leaves the statement text stable for
        # the snapshot regression (which captured aligned columns verbatim).
        # Empty paragraphs are dropped.
        statements = (re.sub(r"\s*\n\s*", " ", p).strip()
                      for p in re.split(r"\n\s*\n", program))
        return [s for s in statements if s]

    def tokenize(self, text: str) -> _TokenizedText:
        """Tokenize a single DecL line.

        Returns an iterable of token objects with ``.type``, ``.value``, and
        ``.index`` attributes (the index is the character offset in the source
        line), compatible with the legacy SLY token shape that
        ``aggregate.underwriter`` inspects on parse errors.
        """
        return _TokenizedText(text)


# ======================================================================
# Transformer (parse tree -> (kind, name, spec) tuple)
# ======================================================================


def _check_vectorizable(value):
    """Coerce a value into something numpy can broadcast over."""
    if isinstance(value, (float, int, np.ndarray)):
        return value
    return np.array(value)


class _PercentNumber(float):
    """Float that remembers it was written with a trailing ``%``.

    Used by ``reins_clause_share`` / ``reins_clause_part`` to
    distinguish a percentage share (``50%``) from an absolute amount
    (``5``) in ``so`` / ``po`` reinsurance clauses. Arithmetic on a
    ``_PercentNumber`` produces a plain ``float`` (the % marker only
    survives literal use), so an expression like ``25 * 2 %`` doesn't
    sneak through as a percentage.
    """
    __slots__ = ()


def _number_to_float(s: str):
    if s.endswith("%"):
        return _PercentNumber(float(s[:-1]) / 100)
    if s == "inf":
        return np.inf
    if s == "-inf":
        return -np.inf
    return float(s)


class UnderwritingTransformer(Transformer):
    """Transform a Lark parse tree into the SLY-compatible
    ``(kind, name, spec)`` tuple consumed by ``aggregate.underwriter``."""

    def __init__(self, safe_lookup_function, debug: bool = False) -> None:
        super().__init__()
        self.safe_lookup = safe_lookup_function
        self.debug = debug

    # ----- terminals -------------------------------------------------
    def NUMBER(self, tok):
        return _number_to_float(str(tok))

    def NOTE(self, tok):
        return str(tok)[5:-1]

    def HINTS(self, tok):
        # strip the leading ``hints{`` (6 chars) and trailing ``}``.
        return str(tok)[6:-1]

    def ID(self, tok):
        return str(tok)

    def BUILTIN_AGG(self, tok):
        return str(tok)

    def BUILTIN_SEV(self, tok):
        return str(tok)

    def BUILTIN_DIST(self, tok):
        """Normalise ``dist.X`` and ``distortion.X`` to ``distortion.X``
        so :meth:`Underwriter._safe_lookup` finds the entry under its
        canonical kind."""
        s = str(tok)
        if s.startswith('dist.'):
            s = 'distortion.' + s[len('dist.'):]
        return s

    def FREQ(self, tok):
        return str(tok)

    # ----- answer dispatch ------------------------------------------
    def answer_sev(self, c):
        return c[0]

    def answer_agg(self, c):
        return c[0]

    def answer_port(self, c):
        return c[0]

    def answer_distortion(self, c):
        return c[0]

    def answer_expr(self, c):
        e = c[0]
        return ("expr", f"{e}", e)

    # ----- distortion ------------------------------------------------
    def distortion_out_params(self, c):
        """``distortion NAME kind n1 n2 ...`` -> ``(kind, name, spec)``.

        The flat parameter list is mapped onto the kind's natural keyword
        arguments by :meth:`aggregate.spectral.Distortion.decl_spec`; the
        per-kind ordering lives on the Distortion subclass (``decl_params``),
        not here, so the parser holds no distortion-specific knowledge."""
        from .spectral import Distortion

        _, name, kind_id, numbers = c
        return ("distortion", name, Distortion.decl_spec(kind_id, numbers))

    def buildin_dist_list_one(self, c):
        return [c[0]]

    def buildin_dist_list_cons(self, c):
        lst, tok = c
        lst.append(tok)
        return lst

    def _resolve_combo_children(self, ids):
        """Resolve a list of ``distortion.X`` ids to actual Distortion
        instances by looking each up in the knowledge and constructing it
        from its stored spec."""
        # Local import: spectral imports nothing parser-related, but the
        # parser is imported during aggregate package init before
        # ``Distortion`` is bound at module level. Inline is safe.
        from .spectral import Distortion
        children = []
        for buildinid in ids:
            spec = self.safe_lookup(buildinid)
            children.append(Distortion(**spec))
        return children

    def distortion_out_combo(self, c):
        _, name, kind_id, child_ids = c
        if kind_id not in ('minimum', 'mixture'):
            raise ValueError(
                f"DecL: '{kind_id}' does not take a list of distortion "
                f"references; only 'minimum' and 'mixture' do")
        children = self._resolve_combo_children(child_ids)
        return ("distortion", name,
                {"name": kind_id, "distortions": children})

    def distortion_out_combo_wtd(self, c):
        _, name, kind_id, child_ids, _wts_kw, wts = c
        if kind_id != 'mixture':
            raise ValueError(
                f"DecL: weights are only meaningful for 'mixture', not "
                f"{kind_id!r}")
        children = self._resolve_combo_children(child_ids)
        return ("distortion", name,
                {"name": kind_id, "distortions": children, "wts": wts})

    # ----- portfolio -------------------------------------------------
    def port_out(self, c):
        _, name, trailer, agg_list = c
        return ("port", name, {"spec": agg_list,
                               "note": trailer["note"],
                               "hints": trailer["hints"]})

    def agg_list_cons(self, c):
        lst, ag = c
        lst.append(ag)
        return lst

    def agg_list_one(self, c):
        return [c[0]]

    # ----- aggregate -------------------------------------------------
    def agg_out_full(self, c):
        _, name, exposures, layers, sev_clause, occ_reins, freq, agg_reins, approx, trailer = c
        spec = {
            "name": name,
            **exposures,
            **layers,
            **sev_clause,
            **occ_reins,
            **freq,
            **agg_reins,
            **self._check_approx(approx, occ_reins),
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("agg", name, spec)

    def agg_out_dfreq(self, c):
        _, name, dfreq, layers, sev_clause, occ_reins, agg_reins, approx, trailer = c
        spec = {
            "name": name,
            **dfreq,
            **layers,
            **sev_clause,
            **occ_reins,
            **agg_reins,
            **self._check_approx(approx, occ_reins),
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("agg", name, spec)

    def agg_out_tweedie(self, c):
        # Tweedie distribution in (mean, p, sigma^2) form. The variance
        # function is sigma^2 * mean^p; phi = sigma^2 in Jorgenson p. 127
        # notation. The Tweedie -> compound-Poisson(gamma) reparameterization
        # is delegated to ``tweedie_convert`` (imported lazily because this
        # module is also runnable as ``python -m`` for grammar printing).
        from .tweedie import tweedie_convert

        _, name, _tw, mu, pp, sig2, trailer = c
        ans = tweedie_convert(p=pp, μ=mu, σ2=sig2)
        alpha = ans["α"]
        lam = ans["λ"]
        beta = ans["β"]
        spec = {
            "name": name,
            "exp_en": lam,
            "freq_name": "poisson",
            "sev_name": "gamma",
            "sev_a": alpha,
            "sev_scale": beta,
            # tweedie synthesises its own descriptive note (the user note is
            # not preserved, as before); hints still flow through.
            "note": (
                f"Tw(p={pp}, μ={mu}, σ^2={sig2}) --> "
                f"CP(λ={lam:8g}, ga(α={alpha:.8g}, β={beta:.8g}), scale={beta:.8g}"
            ),
            "hints": trailer["hints"],
        }
        return ("agg", name, spec)

    def agg_out_rename(self, c):
        _, name, bagg, occ_reins, agg_reins, trailer = c
        if "name" in bagg:
            del bagg["name"]
        spec = {"name": name, **bagg, **occ_reins, **agg_reins,
                "note": trailer["note"], "hints": trailer["hints"]}
        return ("agg", name, spec)

    def agg_out_builtin(self, c):
        bagg, agg_reins, trailer = c
        return ("agg", bagg["name"], {**bagg, **agg_reins,
                                      "note": trailer["note"],
                                      "hints": trailer["hints"]})

    # ----- profit-and-loss aggregate (premium minus loss) -----------
    def answer_pnl(self, c):
        return c[0]

    def _attach_pnl(self, spec, premium):
        """Attach the premium-minus-loss affine wrapper to a loss ``spec``.

        ``pnl`` builds the same spec an ``agg`` would for the loss body, then
        records an aggregate-level affine transform: reflect (a profit is a
        negative loss) and a single deterministic shift equal to the total
        premium. The premium is subtracted **once for the book**, in contrast
        to a constant inside ``sev``/``dsev``/``ssev`` which is per-claim.

        Parameters
        ----------
        spec : dict
            The loss-aggregate spec (mutated in place).
        premium : float or list
            The stated premium (scalar or per-component vector). The shift is
            the sum; ``value_type`` is set to ``'payoff'`` (more is better).

        Notes
        -----
        For the bare-``lr`` exposure form (``_pnl_lr`` marker on the spec) the
        premium also drives the loss ratio: ``E[loss] = premium * lr``,
        synthesised here exactly as :meth:`exposures_premium_lr` does, so the
        per-component claim-count derivation is reused unchanged.
        """
        if "_pnl_lr" in spec:
            lr = spec.pop("_pnl_lr")
            spec["exp_premium"] = premium
            spec["exp_lr"] = lr
            spec["exp_el"] = np.array(premium) * np.array(lr)
        shift = float(np.sum(np.asarray(_check_vectorizable(premium), dtype=float)))
        spec["agg_premium"] = premium
        spec["agg_reflect"] = True
        spec["agg_shift"] = shift
        spec["value_type"] = "payoff"

    def pnl_out_full(self, c):
        (_pnl, name, premium, _prem, _minus, exposures, layers, sev_clause,
         occ_reins, freq, agg_reins, approx, trailer) = c
        spec = {
            "name": name,
            **exposures,
            **layers,
            **sev_clause,
            **occ_reins,
            **freq,
            **agg_reins,
            **self._check_approx(approx, occ_reins),
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        self._attach_pnl(spec, premium)
        return ("agg", name, spec)

    def pnl_out_dfreq(self, c):
        (_pnl, name, premium, _prem, _minus, dfreq, layers, sev_clause,
         occ_reins, agg_reins, approx, trailer) = c
        spec = {
            "name": name,
            **dfreq,
            **layers,
            **sev_clause,
            **occ_reins,
            **agg_reins,
            **self._check_approx(approx, occ_reins),
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        self._attach_pnl(spec, premium)
        return ("agg", name, spec)

    def pnl_exp_claims(self, c):
        numbers, _claims = c
        return {"exp_en": numbers}

    def pnl_exp_loss(self, c):
        numbers, _loss = c
        return {"exp_el": numbers}

    def pnl_exp_lr(self, c):
        # Bare loss ratio: binds to the pnl premium (resolved in _attach_pnl,
        # which has the premium in scope). ``lr`` is just the multiplier --
        # no expense / combined-ratio meaning.
        lr, _lr = c
        return {"_pnl_lr": _check_vectorizable(lr)}

    # ----- bivariate (copula-coupled) -------------------------------
    def answer_bv(self, c):
        return c[0]

    def bv_body_one(self, c):
        return [c[0]]

    def bv_body_cons(self, c):
        lst, item = c
        lst.append(item)
        return lst

    def copula_one_param(self, c):
        """Build the :class:`Copula` from ``copula KIND P``.

        ``KIND`` is an id (e.g. ``gumbel``); ``P`` is the kind's natural
        parameter (Kendall tau, Pearson rho, or Spearman rho_s -- per the
        :class:`aggregate.copula.Copula` subclass)."""
        from .copula import Copula

        _copula, kind_id, param = c
        return Copula(kind_id, float(param))

    def copula_no_param(self, c):
        """Build a parameter-free :class:`Copula` from ``copula KIND`` (e.g.
        ``copula independent``)."""
        from .copula import Copula

        _copula, kind_id = c
        return Copula(kind_id)

    def copula_none(self, c):
        """No copula clause -> the independence copula (the default)."""
        from .copula import Copula

        return Copula('independent')

    def bv_out_copula(self, c):
        _mv, name, exposures, body, copula, freq, trailer = c
        spec = {
            "name": name,
            **exposures,
            **freq,
            "units": body,
            "copula": copula,
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    def bv_out_copula_nofreq(self, c):
        _mv, name, exposures, body, copula, trailer = c
        spec = {
            "name": name,
            **exposures,
            "freq_name": "poisson",
            "units": body,
            "copula": copula,
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    def bv_out_copula_dfreq(self, c):
        """``bivariate NAME dfreq [...] [...] <two aggs> copula ...`` (form 2).

        The ``dfreq`` clause carries both the shared event count and its
        distribution (an empirical frequency), replacing the ``exposures ...
        freq`` head exactly as in ``agg_out_dfreq``; the rest is the copula
        builder.
        """
        _bv, name, dfreq, body, copula, trailer = c
        spec = {
            "name": name,
            **dfreq,
            "units": body,
            "copula": copula,
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    def bv_out_discrete(self, c):
        """``bivariate NAME <count> dbvsev ... <freq>`` (form 3).

        A discrete bivariate severity (the joint per-claim matrix given directly)
        with the full frequency vocabulary after it (bare count -> Poisson, or any
        named distribution); ``mode='discrete'``.
        """
        _bv, name, exposures, dbv, freq, trailer = c
        spec = {
            "name": name,
            "mode": "discrete",
            **exposures,
            **freq,
            **dbv,
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    def bv_out_discrete_nofreq(self, c):
        """``bivariate NAME <count> dbvsev ...`` with no trailing freq -> Poisson."""
        _bv, name, exposures, dbv, trailer = c
        spec = {
            "name": name,
            "mode": "discrete",
            **exposures,
            "freq_name": "poisson",
            **dbv,
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    def bv_out_discrete_dfreq(self, c):
        """``bivariate NAME dfreq [...] [...] dbvsev ...`` (form 4, the headline).

        Both the shared frequency and the joint per-claim severity are discrete:
        a ``dfreq`` empirical count and a ``dbvsev`` lattice; ``mode='discrete'``.
        """
        _bv, name, dfreq, dbv, trailer = c
        spec = {
            "name": name,
            "mode": "discrete",
            **dfreq,
            **dbv,
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    @staticmethod
    def _bv_out_viewpair(c, views):
        """Shared builder for the three occurrence view-pair prefixes.

        ``<keyword> <agg with occurrence reinsurance>`` -> the joint
        per-occurrence aggregate of the named pair of {gross, ceded, net} as a
        ``netceded``-mode BivariateAggregate. ``views`` is the ``(x, y)`` axis
        pair the keyword names (e.g. ``('net', 'ceded')`` for ``netceded``).
        """
        _kw, agg_tuple = c          # agg_tuple = ("agg", name, spec)
        _, name, spec = agg_tuple
        return ("bvagg", name, {
            "name": name,
            "mode": "netceded",
            "nc_views": views,
            "units": [agg_tuple],
            "note": spec.get("note", ""),
            "hints": spec.get("hints", ""),
        })

    def bv_out_netceded(self, c):
        """``netceded <agg>`` -> joint (x=net, y=ceded) occurrence aggregate."""
        return self._bv_out_viewpair(c, ('net', 'ceded'))

    def bv_out_grossceded(self, c):
        """``grossceded <agg>`` -> joint (x=gross, y=ceded) occurrence aggregate."""
        return self._bv_out_viewpair(c, ('gross', 'ceded'))

    def bv_out_grossnet(self, c):
        """``grossnet <agg>`` -> joint (x=gross, y=net) occurrence aggregate."""
        return self._bv_out_viewpair(c, ('gross', 'net'))

    # ----- clash (independent-trigger shared-event model) ------------
    def clash_comp(self, c):
        """A clash component body: ``layers sev_clause`` -> a partial spec dict."""
        layers, sev_clause = c
        return {**layers, **sev_clause}

    def _clash_spec(self, name, na, nb, nc, comp_a, comp_b, freq, trailer):
        """Build the clash bivariate spec from the (na, nb, nc) counts.

        The solver (:func:`aggregate.bivariate.solve_clash_model`) turns the
        three counts into the shared event count ``n`` and the two per-event
        trigger probabilities ``pa`` / ``pb``; the two components become
        ``dfreq [0 1] [1-p p]`` Bernoulli factories wrapping the given
        limit/severity clauses, coupled by the **independent** copula on the
        shared frequency ``freq``.
        """
        from .bivariate import solve_clash_model
        from .copula import Copula

        sol = solve_clash_model(na, nb, nc)

        def _component(suffix, body, p):
            spec = {
                "name": f"{name}.{suffix}",
                "freq_name": "empirical",
                "freq_a": np.array([0.0, 1.0]),
                "freq_b": np.array([1.0 - p, p]),
                "exp_en": -1,
                **body,
                "note": "",
                "hints": "",
            }
            return ("agg", spec["name"], spec)

        spec = {
            "name": name,
            "mode": "copula",
            **freq,
            "exp_en": sol.n,
            "units": [_component("A", comp_a, sol.pa),
                      _component("B", comp_b, sol.pb)],
            "copula": Copula("independent"),
            "clash": {"na": sol.na, "nb": sol.nb, "nc": sol.nc,
                      "n0": sol.n0, "pa": sol.pa, "pb": sol.pb},
            "note": trailer["note"],
            "hints": trailer["hints"],
        }
        return ("bvagg", name, spec)

    def clash_out(self, c):
        """``clash NAME na nb nc claims <A> <B> <freq>`` -> clash bivariate."""
        _clash, name, na, nb, nc, _claims, comp_a, comp_b, freq, trailer = c
        return self._clash_spec(name, na, nb, nc, comp_a, comp_b, freq, trailer)

    def clash_out_nofreq(self, c):
        """``clash`` with no trailing frequency clause -> shared poisson."""
        _clash, name, na, nb, nc, _claims, comp_a, comp_b, trailer = c
        return self._clash_spec(name, na, nb, nc, comp_a, comp_b,
                                {"freq_name": "poisson"}, trailer)

    # ----- severity output ------------------------------------------
    def sev_out_sev(self, c):
        _, name, sev, trailer = c
        sev["name"] = name
        sev["note"] = trailer["note"]
        sev["hints"] = trailer["hints"]
        return ("sev", name, sev)

    def sev_out_dsev(self, c):
        _, name, dsev, trailer = c
        dsev["name"] = name
        dsev["note"] = trailer["note"]
        dsev["hints"] = trailer["hints"]
        return ("sev", name, dsev)

    # ----- frequency -------------------------------------------------
    def freq_zm(self, c):
        freq, _zm, expr = c
        freq["freq_zm"] = True
        freq["freq_p0"] = expr
        return freq

    def freq_zt(self, c):
        freq, _zt = c
        freq["freq_zm"] = True
        freq["freq_p0"] = 0.0
        return freq

    def freq_mixed_two(self, c):
        _mixed, id_, a, b = c
        return {"freq_name": id_, "freq_a": a, "freq_b": b}

    def freq_mixed_one(self, c):
        _mixed, id_, a = c
        return {"freq_name": id_, "freq_a": a}

    def freq_two(self, c):
        freq, a, b = c
        if freq != "pascal":
            logger.warning(f"Illogical choice of frequency {freq}, expected pascal")
        return {"freq_name": freq, "freq_a": a, "freq_b": b}

    def freq_one(self, c):
        freq, a = c
        if freq not in ["binomial", "neyman", "neymana", "neymanA", "negbin"]:
            logger.warning(
                f"Illogical choice of frequency {freq}, expected binomial or neyman A"
            )
        return {"freq_name": freq, "freq_a": a}

    def freq_zero(self, c):
        freq = c[0]
        if freq not in ("poisson", "bernoulli", "fixed", "geometric", "logarithmic"):
            logger.error(
                f"Illogical choice for FREQ {freq}, should be poisson, bernoulli, "
                "geometric, logarithmic or fixed."
            )
        return {"freq_name": freq}

    # ----- reinsurance ----------------------------------------------
    def agg_reins_net(self, c):
        return {"agg_reins": c[3], "agg_kind": "net of"}

    def agg_reins_ceded(self, c):
        return {"agg_reins": c[3], "agg_kind": "ceded to"}

    def agg_reins_none(self, c):
        return {}

    def occ_reins_net(self, c):
        return {"occ_reins": c[3], "occ_kind": "net of"}

    def occ_reins_ceded(self, c):
        return {"occ_reins": c[3], "occ_kind": "ceded to"}

    def occ_reins_none(self, c):
        return {}

    # ----- approximate (method-of-moments) directive ----------------
    _APPROX_KINDS = ("exact", "sgamma", "slognorm")

    def approx_set(self, c):
        """``approximate KIND`` -> ``{'approximate': KIND}`` (kind validated)."""
        _approx, kind = c
        kind = str(kind)
        if kind not in self._APPROX_KINDS:
            raise ValueError(
                f"DecL: approximate '{kind}' is not recognised; "
                f"use one of {', '.join(self._APPROX_KINDS)}")
        return {"approximate": kind}

    def approx_none(self, c):
        """Omitted ``approximate`` clause -> no spec key (constructor default)."""
        return {}

    def _check_approx(self, approx, occ_reins):
        """Validate the approximate/occurrence-reinsurance combination.

        The method-of-moments fit replaces the freq x sev convolution, so
        per-occurrence reinsurance -- which acts on the severity *before* that
        convolution -- has nothing to bite on. Reject the combination with a
        clear parse-time error. ``approximate exact`` (the inert default) and
        aggregate reinsurance are always fine.

        Parameters
        ----------
        approx : dict
            ``{'approximate': KIND}`` or ``{}`` from the approx clause.
        occ_reins : dict
            ``{'occ_reins': ..., 'occ_kind': ...}`` or ``{}`` from occ_reins.

        Returns
        -------
        dict
            ``approx`` unchanged (so it can be spread into the spec).
        """
        if approx.get("approximate", "exact") != "exact" and "occ_reins" in occ_reins:
            raise ValueError(
                "DecL: approximate is incompatible with occurrence reinsurance "
                "(the method-of-moments fit bypasses the per-occurrence "
                "convolution); use aggregate reinsurance instead.")
        return approx

    def reins_list_cons(self, c):
        lst, _and, clause = c
        lst.append(clause)
        return lst

    def reins_list_one(self, c):
        return [c[0]]

    def reins_list_tower(self, c):
        tower = c[0]
        limit, attach = tower[0], tower[1]
        return [(1.0, l, a) for l, a in zip(limit, attach)]

    def reins_clause_xs(self, c):
        limit, _xs, attach = c
        return (1.0, limit, attach)

    def reins_clause_share(self, c):
        # ``so`` and ``po`` are synonyms; meaning is set by the leading
        # quantity: a literal percentage (``50%``) is the share
        # directly, a bare number is an absolute amount and the share
        # is ``amount / limit``. The ``_PercentNumber`` carries the
        # ``%``-suffix marker through the parse so this branch can
        # decide. The canonical / PIR usage is ``%`` with ``so`` and
        # absolute with ``po`` — both forms now work either way.
        n, _so, limit, _xs, attach = c
        if isinstance(n, _PercentNumber):
            return (float(n), limit, attach)
        return (n / limit, limit, attach)

    def reins_clause_of(self, c):
        # ``of`` is a natural-language synonym for ``so`` (share of): a
        # literal percentage (``90%``) is the share directly, a bare number
        # is an absolute amount and the share is ``amount / limit``. Reads as
        # a share, so -- unlike ``po`` -- no "suspiciously small" warning.
        n, _of, limit, _xs, attach = c
        if isinstance(n, _PercentNumber):
            return (float(n), limit, attach)
        return (n / limit, limit, attach)

    def reins_clause_part(self, c):
        n, _po, limit, _xs, attach = c
        if isinstance(n, _PercentNumber):
            return (float(n), limit, attach)
        if n / limit < 0.05:
            logger.warning(
                f"Part of clause with proportion {n / limit} is "
                "suspiciously small. Did you mean share of?"
            )
        return (n / limit, limit, attach)

    # ----- severity (continuous) ------------------------------------
    def sev_clause_sev(self, c):
        _sev, sev = c
        return sev

    def sev_clause_ssev(self, c):
        # ssev = signed (never-clamp) continuous severity: a profit is a
        # negative loss. Same spec as sev, flagged so the Severity keeps its
        # negative support instead of clamping x<0 -> 0. Orthogonal to
        # value_type (does NOT imply payoff). See dev/plan-negative-x-agg.md.
        _ssev, sev = c
        sev['sev_signed'] = True
        return sev

    def sev_clause_dsev(self, c):
        return c[0]

    def sev_clause_builtin(self, c):
        b = self.safe_lookup(c[0])
        if "name" in b:
            del b["name"]
        return b

    def sev_unconditional(self, c):
        sev = c[0]
        sev["sev_conditional"] = False
        return sev

    def sev_picks(self, c):
        sev, picks = c
        return {**sev, **picks}

    def sev_weighted(self, c):
        sev2, weights, splice = c
        sev2["sev_wt"] = weights
        sev2["sev_lb"] = splice["sev_lb"]
        sev2["sev_ub"] = splice["sev_ub"]
        return sev2

    def sev_builtin(self, c):
        b = self.safe_lookup(c[0])
        if "name" in b:
            del b["name"]
        return b

    def sev2_add(self, c):
        sev1, _plus, numbers = c
        sev1["sev_loc"] = _check_vectorizable(sev1.get("sev_loc", 0))
        sev1["sev_loc"] += _check_vectorizable(numbers)
        return sev1

    def sev2_sub(self, c):
        sev1, _minus, numbers = c
        sev1["sev_loc"] = _check_vectorizable(sev1.get("sev_loc", 0))
        sev1["sev_loc"] -= _check_vectorizable(numbers)
        return sev1

    def sev2_rsub(self, c):
        """``shift - X``: a constant minus a distribution (e.g. premium minus
        loss). The natural reading of a profit/loss severity. Equivalent to
        ``-1 * X + shift`` (so it requires ``ssev`` to keep the signed support;
        under plain ``sev`` the sub-zero tail clamps as usual).

        ``X`` (``sev1``) has value ``Lx + s*base`` with ``s = -1`` if already
        reflected else ``+1``. Then ``shift - X = (shift - Lx) + (-s)*base``,
        so set ``sev_loc = shift - Lx`` and toggle ``sev_reflect``. ``loc`` is
        applied additively *after* reflection in ``Severity`` (independent of
        the reflect sign), matching the ``sev1_scaled`` convention.
        """
        numbers, _minus, sev1 = c
        shift = _check_vectorizable(numbers)
        lx = _check_vectorizable(sev1.get("sev_loc", 0))
        sev1["sev_loc"] = shift - lx
        sev1["sev_reflect"] = not sev1.get("sev_reflect", False)
        return sev1

    def sev2_passthrough(self, c):
        return c[0]

    def sev1_scaled(self, c):
        numbers, _times, sev0 = c
        p_numbers = _check_vectorizable(numbers)
        # A negative multiplier means reflection (``-X``). scipy cannot carry a
        # negative scale, so record a ``sev_reflect`` flag (toggled, so two
        # negatives cancel) and scale by the magnitude; the Severity builds the
        # positive distribution and reflects it (signed support). The trailing
        # ``+/- shift`` (sev2) is then applied as ``shift - X``.
        if np.any(np.asarray(p_numbers) < 0):
            sev0["sev_reflect"] = not sev0.get("sev_reflect", False)
        mag = np.abs(p_numbers)
        if "sev_mean" in sev0:
            sev0["sev_mean"] = _check_vectorizable(sev0.get("sev_mean", 0)) * mag
        if "sev_scale" in sev0:
            sev0["sev_scale"] = (
                _check_vectorizable(sev0.get("sev_scale", 0)) * mag
            )
        if "sev_mean" not in sev0:
            # Distributions without an analytic mean (e.g. Pareto) get a scale
            # rather than a scaled mean; setting both would double-count.
            sev0["sev_scale"] = mag
        if "sev_loc" in sev0:
            sev0["sev_loc"] = _check_vectorizable(sev0["sev_loc"]) * mag
        return sev0

    def sev1_passthrough(self, c):
        return c[0]

    def sev0_mean_cv(self, c):
        ids, mean, _cv, cv = c
        return {"sev_name": ids, "sev_mean": mean, "sev_cv": cv, "sev_scale": 1.0}

    def sev0_two_params(self, c):
        ids, a, b = c
        return {"sev_name": ids, "sev_a": a, "sev_b": b, "sev_scale": 1.0}

    def sev0_one_param(self, c):
        ids, a = c
        return {"sev_name": ids, "sev_a": a, "sev_scale": 1.0}

    def sev0_xps(self, c):
        ids, xps = c
        return {"sev_name": ids, **xps}

    def sev0_zero_params(self, c):
        ids = c[0]
        return {"sev_name": ids, "sev_scale": 1.0}

    def xps(self, c):
        _xps, doutcomes, dprobs = c
        ps = np.ones_like(doutcomes) / len(doutcomes) if len(dprobs) == 0 else dprobs
        return {"sev_xs": doutcomes, "sev_ps": ps}

    def dsev_main(self, c):
        _dsev, doutcomes, dprobs = c
        ps = np.ones_like(doutcomes) / len(doutcomes) if len(dprobs) == 0 else dprobs
        return {"sev_name": "dhistogram", "sev_xs": doutcomes, "sev_ps": ps}

    def dsev_unconditional(self, c):
        dsev = c[0]
        dsev["sev_conditional"] = False
        return dsev

    def dfreq(self, c):
        _dfreq, doutcomes, dprobs = c
        b = np.ones_like(doutcomes) / len(doutcomes) if len(dprobs) == 0 else dprobs
        return {
            "freq_name": "empirical",
            "freq_a": doutcomes,
            "freq_b": b,
            "exp_en": -1,
        }

    # ----- dbvsev (discrete bivariate severity) ---------------------
    # Every dbvsev surface form (dense / dense-uniform / sparse) normalises to
    # the SAME partial spec ``{dbv_xs, dbv_ys, dbv_S}`` -- the joint per-claim
    # probability matrix on an explicit lattice -- so BivariateAggregate sees one
    # shape regardless of how it was written. See dev/done/plan-bv-discrete.md.
    @staticmethod
    def _finalize_dbv(xs, ys, S):
        """Validate / renormalise a dbvsev lattice into ``{dbv_xs, dbv_ys, dbv_S}``.

        Checks the matrix shape against the axis lengths and non-negativity, then
        renormalises to sum 1 (warning if off by more than rounding noise), the
        2-D analogue of ``dsev``'s pmf handling.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        S = np.asarray(S, dtype=float)
        if S.shape != (len(xs), len(ys)):
            raise ValueError(
                f"dbvsev: probability matrix shape {S.shape} does not match the "
                f"lattice ({len(xs)} x outcomes, {len(ys)} y outcomes); expected "
                f"{(len(xs), len(ys))}.")
        if np.any(S < 0):
            raise ValueError("dbvsev: probability matrix has negative entries.")
        total = float(S.sum())
        if total <= 0:
            raise ValueError("dbvsev: probability matrix sums to zero.")
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "dbvsev: probabilities sum to %.6g, renormalising to 1.", total)
        return {"dbv_xs": xs, "dbv_ys": ys, "dbv_S": S / total}

    def drow(self, c):
        return _check_vectorizable(c[0])

    def dmatrix_rows_one(self, c):
        return [c[0]]

    def dmatrix_rows_cons(self, c):
        rows, row = c
        rows.append(row)
        return rows

    def dprob_matrix(self, c):
        # c[0] is the list of 1-D row arrays; ragged rows raise in _finalize_dbv.
        return np.array([np.asarray(r, dtype=float) for r in c[0]])

    def dbvsev_dense(self, c):
        _dbvsev, xs, ys, matrix = c
        return self._finalize_dbv(xs, ys, matrix)

    def dbvsev_dense_uniform(self, c):
        _dbvsev, xs, ys = c
        xs = _check_vectorizable(xs)
        ys = _check_vectorizable(ys)
        nx, ny = len(xs), len(ys)
        return self._finalize_dbv(xs, ys, np.full((nx, ny), 1.0 / (nx * ny)))

    def dtriple(self, c):
        x, y, p = c
        return (float(x), float(y), float(p))

    def dtriples_one(self, c):
        return [c[0]]

    def dtriples_cons(self, c):
        lst, t = c
        lst.append(t)
        return lst

    def dtriple_list(self, c):
        return c[0]

    def dbvsev_sparse(self, c):
        _dbvsev, triples = c
        xs = sorted({t[0] for t in triples})
        ys = sorted({t[1] for t in triples})
        ix = {v: i for i, v in enumerate(xs)}
        iy = {v: j for j, v in enumerate(ys)}
        S = np.zeros((len(xs), len(ys)))
        for x, y, p in triples:
            S[ix[x], iy[y]] += p          # collisions summed
        return self._finalize_dbv(xs, ys, S)

    def picks(self, c):
        _picks, attachments, losses = c
        return {"sev_pick_attachments": attachments, "sev_pick_losses": losses}

    def doutcomes_list(self, c):
        return _check_vectorizable(c[0])

    def doutcomes_range(self, c):
        start, _, end = c
        return np.arange(start, end + 1)

    def doutcomes_range_step(self, c):
        start, _, end, _2, step = c
        return np.arange(start, end + 0.5 * step, step)

    def dprobs_list(self, c):
        return _check_vectorizable(c[0])

    def dprobs_none(self, c):
        return []

    def weights_equal(self, c):
        _wts, _eq, expr = c
        return np.ones(int(expr)) / expr

    def weights_list(self, c):
        _wts, numberl = c
        return numberl

    def weights_none(self, c):
        return 1.0

    def splice_two(self, c):
        _, lb, ub = c
        return {"sev_lb": lb, "sev_ub": ub}

    def splice_one(self, c):
        _, numberl = c
        return {"sev_lb": numberl[:-1], "sev_ub": numberl[1:]}

    def splice_none(self, c):
        return {"sev_lb": 0.0, "sev_ub": np.inf}

    # ----- layers ----------------------------------------------------
    def layers_xs(self, c):
        limit, _xs, attach = c
        return {"exp_attachment": attach, "exp_limit": limit}

    def layers_tower(self, c):
        tower = c[0]
        return {"exp_attachment": tower[1], "exp_limit": tower[0]}

    def layers_none(self, c):
        return {}

    def tower(self, c):
        _tower, doutcomes = c
        breaks = doutcomes
        limits = np.diff(breaks)
        attach = breaks[:-1]
        return [limits, attach]

    # ----- trailer (optional note{...} + hints{...}) -----------------
    # Each method returns ``{"note": <text>, "hints": <raw settings string>}``
    # so the order-free / present-or-absent variants collapse to one shape.
    # ``hints`` is left as a raw ``key=value;`` string here; it is parsed and
    # type-coerced in ``aggregate.underwriter`` (caller-wins merge).
    def trailer_nh(self, c):
        return {"note": c[0], "hints": c[1]}

    def trailer_hn(self, c):
        return {"note": c[1], "hints": c[0]}

    def trailer_note(self, c):
        return {"note": c[0], "hints": ""}

    def trailer_hints(self, c):
        return {"note": "", "hints": c[0]}

    def trailer_none(self, c):
        return {"note": "", "hints": ""}

    # ----- exposures -------------------------------------------------
    def exposures_claims(self, c):
        numbers, _claims = c
        return {"exp_en": numbers}

    def exposures_loss(self, c):
        numbers, _loss = c
        return {"exp_el": numbers}

    def exposures_premium_lr(self, c):
        prem, _premium, _at, lr, _lr = c
        return {
            "exp_premium": prem,
            "exp_lr": lr,
            "exp_el": np.array(prem) * np.array(lr),
        }

    def exposures_exposure_rate(self, c):
        exp_, _exposure, _at, rate, _rate = c
        return {
            "exp_premium": exp_,
            "exp_lr": rate,
            "exp_el": np.array(exp_) * np.array(rate),
        }

    # ----- ids -------------------------------------------------------
    def ids_list(self, c):
        return c[0]

    def ids_single(self, c):
        return c[0]

    def idl_cons(self, c):
        lst, id_ = c
        lst.append(id_)
        return lst

    def idl_one(self, c):
        return [c[0]]

    # ----- builtin aggregate scaling --------------------------------
    def builtin_agg_inhomog(self, c):
        expr, _at, bagg = c
        bid = bagg.copy()
        bid["name"] += "_i_scaled"
        bid["exp_en"] = _check_vectorizable(bid.get("exp_en", 0)) * expr
        bid["exp_el"] = _check_vectorizable(bid.get("exp_el", 0)) * expr
        bid["exp_premium"] = _check_vectorizable(bid.get("exp_premium", 0)) * expr
        return bid

    def builtin_agg_homog(self, c):
        expr, _times, bagg = c
        bid = bagg
        bid["name"] += "_homog_scaled"
        if "sev_mean" in bid:
            bid["sev_mean"] = _check_vectorizable(bid["sev_mean"]) * expr
        if "sev_scale" in bid:
            bid["sev_scale"] = _check_vectorizable(bid["sev_scale"]) * expr
        if "sev_loc" in bid:
            bid["sev_loc"] = _check_vectorizable(bid["sev_loc"]) * expr
        bid["exp_attachment"] = _check_vectorizable(bid.get("exp_attachment", 0)) * expr
        bid["exp_limit"] = _check_vectorizable(bid.get("exp_limit", np.inf)) * expr
        bid["exp_el"] = _check_vectorizable(bid.get("exp_el", 0)) * expr
        bid["exp_premium"] = _check_vectorizable(bid.get("exp_premium", 0)) * expr
        return bid

    def builtin_agg_plus(self, c):
        bagg, _plus, expr = c
        bid = bagg
        bid["name"] += "_shifted"
        if "sev_loc" in bid:
            bid["sev_loc"] += expr
        else:
            bid["sev_loc"] = expr
        return bid

    def builtin_agg_minus(self, c):
        bagg, _minus, expr = c
        bid = bagg
        bid["name"] += "_shifted"
        if "sev_loc" in bid:
            bid["sev_loc"] -= expr
        else:
            bid["sev_loc"] = -expr
        return bid

    def builtin_agg_lookup(self, c):
        return self.safe_lookup(c[0])

    # ----- name ------------------------------------------------------
    def name(self, c):
        return c[0]

    # ----- numbers (vectors and scalars) -----------------------------
    def numbers_list(self, c):
        return c[0]

    def numbers_range(self, c):
        start, _, end = c
        return np.arange(start, end + 1)

    def numbers_range_step(self, c):
        start, _, end, _2, step = c
        return np.arange(start, end + 1, step)

    def numbers_scalar(self, c):
        return c[0]

    def numberl_cons(self, c):
        lst, expr = c
        lst.append(expr)
        return lst

    def numberl_one(self, c):
        return [c[0]]

    # ----- expressions (DecL math sub-language) ----------------------
    # ?expr / ?term / ?factor are inlined in the grammar — the
    # transformer only sees the aliased nodes below.

    def atom_divide(self, c):
        a, _, b = c
        return a / b

    def atom_parens(self, c):
        return c[0]

    def atom_exp(self, c):
        _, x = c
        return np.exp(x)

    def atom_exponent(self, c):
        a, _, b = c
        return a ** b

    def atom_number(self, c):
        return c[0]


# ======================================================================
# Module-level Lark instance (loaded once)
# ======================================================================

_PARSER = Lark.open(
    str(GRAMMAR_FILE),
    start="answer",
    parser="earley",
    lexer="dynamic",
    maybe_placeholders=True,
)


# ======================================================================
# Parser wrapper
# ======================================================================


class UnderwritingParser:
    """DecL parser. Accepts either a raw line of source or the iterable
    returned by ``UnderwritingLexer.tokenize`` (which carries the original
    text), returning a ``(kind, name, spec)`` tuple where ``kind`` is one of
    ``'agg'``, ``'sev'``, ``'port'``, ``'distortion'``, or ``'expr'``."""

    def __init__(self, safe_lookup_function, debug: bool = False) -> None:
        self.safe_lookup = safe_lookup_function
        self.debug = debug

    def parse(self, source) -> tuple[str, str, object]:
        """Parse a single DecL statement.

        Parameters
        ----------
        source : str or iterable
            Either the source string directly, or a ``_TokenizedText``
            instance (the return value of ``UnderwritingLexer.tokenize``).

        Returns
        -------
        (kind, name, spec) : tuple
            ``kind`` is one of ``'agg'``, ``'sev'``, ``'port'``,
            ``'distortion'``, ``'expr'``; ``name`` is the object identifier;
            ``spec`` is the dictionary specification used downstream to
            construct the object.

        Raises
        ------
        ValueError
            On parse error. ``args[0]`` is a one-line human-readable
            summary (location, unexpected token, "did you mean"
            suggestion). The structured form, with line/column,
            caret-annotated source line, and the full expected-terminal
            set, is attached as ``err.report`` (an
            :class:`~aggregate.parser_errors.ErrorReport`).
            ``err.report.render()`` gives the multi-line text form.
        """
        if isinstance(source, str):
            text = source
        else:
            text = getattr(source, "text", None) or "".join(
                str(t.value) for t in source
            )
        try:
            tree = _PARSER.parse(text)
        except (UnexpectedToken, UnexpectedCharacters, UnexpectedInput) as e:
            report = format_error(text, e)
            err = ValueError(report.summary)
            err.report = report
            raise err from None
        try:
            return UnderwritingTransformer(
                self.safe_lookup, self.debug).transform(tree)
        except VisitError as e:
            # Surface a transformer-raised error (e.g. dbvsev validation) as the
            # original exception rather than Lark's wrapper.
            raise e.orig_exc from None


# ======================================================================
# Documentation helper
# ======================================================================


def grammar(add_to_doc: bool = False, save_to_fn: str | Path = "") -> str:
    """Return the DecL grammar (the contents of ``decl.lark``) as a string.

    Parameters
    ----------
    add_to_doc : bool
        If True, write the grammar to ``docs/4_agg_language_reference/
        ref_include.rst`` wrapped in a Sphinx ``code-block:: lark`` directive
        so it can be ``include``-d by the language reference.
    save_to_fn : str or Path
        Optional additional output path. If empty, defaults to
        ``~/aggregate/parser/grammar.lark``.

    Returns
    -------
    str
        The full grammar source.
    """
    text = GRAMMAR_FILE.read_text(encoding="utf-8")

    if add_to_doc:
        # repo root is three levels up from src/aggregate/parser.py
        # (src/aggregate -> src -> repo root); the docs tree lives at the root.
        out = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "4_agg_language_reference"
            / "ref_include.rst"
        )
        body = "\n".join("    " + ln for ln in text.splitlines())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f".. code-block:: lark\n\n{body}\n", encoding="utf-8")

    target = Path(save_to_fn) if save_to_fn else Path.home() / "aggregate/parser/grammar.lark"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    return text


if __name__ == "__main__":
    grammar(add_to_doc=True)
