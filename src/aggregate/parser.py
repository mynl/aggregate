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
from typing import Iterable, Iterator

import numpy as np
from lark import Lark, Transformer
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken

from .parser_errors import format_error

logger = logging.getLogger(__name__)

__all__ = ['UnderwritingLexer', 'UnderwritingParser', 'grammar']

GRAMMAR_FILE = Path(__file__).parent / "decl.lark"


def _distortion_spec(kind_id, shape, df=None):
    """
    Translate a DecL ``(kind, shape, df?)`` triple into a natural-kwarg
    spec dict ready for ``Distortion(**spec)``.

    Each distortion kind has its own natural parameter names (post the
    1.0.0a13 reform); the DecL grammar still parses the legacy
    ``kind shape [df]`` form, so this helper sits between the parser and
    the constructor and dispatches on the kind id.

    Parameters
    ----------
    kind_id : str
        The kind identifier parsed from DecL.
    shape : object
        The parsed ``shape`` token (float or list, per the kind).
    df : object, optional
        The optional second slot, used by ``bitvar`` and ``wtdtvar``.

    Returns
    -------
    dict
        ``{'name': kind_id, **natural_kwargs}`` to be unpacked into
        ``Distortion(**spec)``.

    Raises
    ------
    ValueError
        If ``df`` is missing for kinds that require it, or the kind id
        is unknown.
    """
    spec = {'name': kind_id}
    if kind_id == 'ph':
        spec['a'] = shape
    elif kind_id == 'wang':
        spec['lam'] = shape
    elif kind_id == 'dual':
        spec['b'] = shape
    elif kind_id == 'tvar':
        spec['p'] = shape
    elif kind_id in ('ccoc', 'roe'):
        # DSL semantics: positional shape is the return r.
        spec['name'] = 'ccoc'
        spec['r'] = shape
    elif kind_id == 'bitvar':
        if df is None or len(df) != 2:
            raise ValueError(
                f'DecL bitvar requires [p0 p1]; got df={df!r}')
        spec['p0'], spec['p1'] = df[0], df[1]
        spec['w1'] = shape
    elif kind_id == 'wtdtvar':
        if df is None:
            raise ValueError(
                f'DecL wtdtvar requires [wts]; got df={df!r}')
        spec['ps'] = shape
        spec['wts'] = df
    elif kind_id == 'cll':
        spec['b'] = shape
    elif kind_id == 'clin':
        spec['slope'] = shape
    elif kind_id == 'lep':
        spec['r'] = shape
    elif kind_id == 'ly':
        spec['r'] = shape
    elif kind_id == 'beta':
        # DSL form: shape is [a, b]
        spec['a'], spec['b'] = shape[0], shape[1]
    elif kind_id == 'power':
        if df is None or len(df) != 2:
            raise ValueError(
                f'DecL power requires [x0 x1]; got df={df!r}')
        spec['x0'], spec['x1'] = df[0], df[1]
        spec['alpha'] = shape
    else:
        raise ValueError(f'DecL: unknown distortion kind {kind_id!r}')
    return spec


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
        """Split a multi-line DecL program into clean single-line statements.

        The preprocessor performs six steps:

        1. Newlines inside ``[ ]`` (e.g., from formatted numpy arrays) are
           collapsed to spaces.
        2. ``//`` and ``#`` comments are removed through end of line.
        3. ``\\\\n`` line continuation is mapped to a space.
        4. ``\\n\\t`` and four-space indented ``\\n    `` (the tabbed Portfolio
           layout) are collapsed to a space.
        5. The result is split on newlines.
        6. Empty lines are dropped.

        Parameters
        ----------
        program : str
            Raw multi-line DecL source.

        Returns
        -------
        list[str]
            Non-empty, stripped DecL statements ready for parsing.
        """
        # Collapse newlines inside [...] (which can appear when a vector is
        # formatted with f'{np.linspace(...)}').
        out_in = re.split(r"\[|\]", program)
        assert len(out_in) % 2  # must be odd
        odd = [t.replace("\n", " ") for t in out_in[1::2]]
        even = out_in[0::2]
        program = " ".join([even[0]] + [f"[{o}] {e}" for o, e in zip(odd, even[1:])])

        # Strip // and # comments through end of line.
        program = re.sub(r"(//|#)[^\n]*$", r"\n", program, flags=re.MULTILINE)

        # Line continuation and Portfolio-indent collapse.
        program = (
            program.replace("\\\n", " ").replace("\n\t", " ").replace("\n    ", " ")
        )

        return [i.strip() for i in program.split("\n") if len(i.strip()) > 0]

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
    def distortion_out_short(self, c):
        _, name, kind_id, shape = c
        spec = _distortion_spec(kind_id, shape)
        return ("distortion", name, spec)

    def distortion_out_long(self, c):
        _, name, kind_id, shape, df = c
        spec = _distortion_spec(kind_id, shape, df)
        return ("distortion", name, spec)

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
        _, name, note, agg_list = c
        return ("port", name, {"spec": agg_list, "note": note})

    def agg_list_cons(self, c):
        lst, ag = c
        lst.append(ag)
        return lst

    def agg_list_one(self, c):
        return [c[0]]

    # ----- aggregate -------------------------------------------------
    def agg_out_full(self, c):
        _, name, exposures, layers, sev_clause, occ_reins, freq, agg_reins, note = c
        spec = {
            "name": name,
            **exposures,
            **layers,
            **sev_clause,
            **occ_reins,
            **freq,
            **agg_reins,
            "note": note,
        }
        return ("agg", name, spec)

    def agg_out_dfreq(self, c):
        _, name, dfreq, layers, sev_clause, occ_reins, agg_reins, note = c
        spec = {
            "name": name,
            **dfreq,
            **layers,
            **sev_clause,
            **occ_reins,
            **agg_reins,
            "note": note,
        }
        return ("agg", name, spec)

    def agg_out_tweedie(self, c):
        # Tweedie distribution in (mean, p, sigma^2) form. The variance
        # function is sigma^2 * mean^p; phi = sigma^2 in Jorgenson p. 127
        # notation. The Tweedie -> compound-Poisson(gamma) reparameterization
        # is delegated to ``tweedie_convert`` (imported lazily because this
        # module is also runnable as ``python -m`` for grammar printing).
        from .tweedie import tweedie_convert

        _, name, _tw, mu, pp, sig2, note = c
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
            "note": (
                f"Tw(p={pp}, μ={mu}, σ^2={sig2}) --> "
                f"CP(λ={lam:8g}, ga(α={alpha:.8g}, β={beta:.8g}), scale={beta:.8g}"
            ),
        }
        return ("agg", name, spec)

    def agg_out_rename(self, c):
        _, name, bagg, occ_reins, agg_reins, note = c
        if "name" in bagg:
            del bagg["name"]
        spec = {"name": name, **bagg, **occ_reins, **agg_reins, "note": note}
        return ("agg", name, spec)

    def agg_out_builtin(self, c):
        bagg, agg_reins, note = c
        return ("agg", bagg["name"], {**bagg, **agg_reins, "note": note})

    # ----- severity output ------------------------------------------
    def sev_out_sev(self, c):
        _, name, sev, note = c
        sev["name"] = name
        sev["note"] = note
        return ("sev", name, sev)

    def sev_out_dsev(self, c):
        _, name, dsev, note = c
        dsev["name"] = name
        dsev["note"] = note
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

    def sev2_passthrough(self, c):
        return c[0]

    def sev1_scaled(self, c):
        numbers, _times, sev0 = c
        p_numbers = _check_vectorizable(numbers)
        if "sev_mean" in sev0:
            sev0["sev_mean"] = _check_vectorizable(sev0.get("sev_mean", 0)) * p_numbers
        if "sev_scale" in sev0:
            sev0["sev_scale"] = (
                _check_vectorizable(sev0.get("sev_scale", 0)) * p_numbers
            )
        if "sev_mean" not in sev0:
            # Distributions without an analytic mean (e.g. Pareto) get a scale
            # rather than a scaled mean; setting both would double-count.
            sev0["sev_scale"] = p_numbers
        if "sev_loc" in sev0:
            sev0["sev_loc"] = _check_vectorizable(sev0["sev_loc"]) * p_numbers
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

    # ----- note ------------------------------------------------------
    def note_some(self, c):
        return c[0]

    def note_none(self, c):
        return ""

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
        return UnderwritingTransformer(self.safe_lookup, self.debug).transform(tree)


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
        out = (
            Path(__file__).parent.parent
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
