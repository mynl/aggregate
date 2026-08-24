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

__all__ = ['UnderwritingLexer', 'UnderwritingParser', 'grammar',
           'INHERIT_PREMIUM', 'DERIVE_PREMIUM']

GRAMMAR_FILE = Path(__file__).parent / "decl.lark"

# ----------------------------------------------------------------------
# note{...} / tags{...} / hints{...} -- the single-line trailer clauses
# ----------------------------------------------------------------------
# Free text: a note is prose written by a human, so it may legitimately
# contain ``#`` (``5# of limit``), ``//``, or
# square brackets (``E[loss] = 85``). Left in place, the preprocessing steps
# treat all three as DecL punctuation: step 2 truncates the note at the ``#``
# or ``//``, and step 3's bracket collapse pads ``[`` and ``]`` with spaces, so
# ``E[loss]=85`` was silently stored as ``E [loss] =85``.
#
# The fix: step 0b lifts each body out and
# substitutes an indexed placeholder whose alphabet (``A-Za-z0-9_``) is inert
# through every later step; the bodies are put back verbatim at the end, once
# the text has been split into statements. An **index**, not an encoding,
# because the substitution is undone inside ``preprocess`` rather than by the
# parser: a real note body could imitate an encoded payload, but nothing can
# imitate a placeholder that is only ever written by the same call that reads
# it.
#
# The terminals (``decl.lark``: ``/note\\{[^}]*\\}/`` and friends) admit any
# character but ``}``, newlines included. The lift deliberately does NOT match
# across a newline: a body spelled over two lines keeps exactly the behavior it
# had before, rather than acquiring a new one here.
_TRAILER_BODY_RE = re.compile(r"\b(note|tags|hints)\{([^}\n]*)\}")

#: A whole-line comment, optional indent then ``#`` or ``//``. The line form of
#: preprocess step 1, for the line-at-a-time scan in ``raw_statements``.
_FULL_LINE_COMMENT_RE = re.compile(r"^[ \t]*(?://|#)")

#: A trailing comment, to end of line. Preprocess step 2, same pattern.
_INLINE_COMMENT_RE = re.compile(r"(//|#)[^\n]*")

class _InheritPremium:
    """Sentinel for ``inherit premium``: copy the engine's technical premium.

    A ``pnl``/``xpnl`` premium head of ``inherit premium`` records this sentinel
    as the consideration at parse time; the underwriter resolves it after the
    engine is built (reading ``Aggregate.exp_premium`` or the accumulated
    portfolio premium), erroring if the engine has no premium.
    """

    __slots__ = ()

    def __repr__(self):
        return "INHERIT_PREMIUM"


#: Singleton :class:`_InheritPremium` sentinel (see :meth:`pnl_premium_inherit`).
INHERIT_PREMIUM = _InheritPremium()


class _DerivePremium:
    """Sentinel for ``derive premium``: the engine premium grossed up for expenses.

    A ``pnl``/``xpnl`` premium head of ``derive premium`` records this sentinel
    as the consideration at parse time; the underwriter resolves it after the
    engine is built. With engine premium T, fixed expense total F and premium
    expense ratio total r from the ``less`` clause, the derived premium is
    ``(T + F) / (1 - r)``, so premium net of expenses returns exactly T. The
    engine having no premium, any loss basis expense, and r at or above one
    are build errors.
    """

    __slots__ = ()

    def __repr__(self):
        return "DERIVE_PREMIUM"


#: Singleton :class:`_DerivePremium` sentinel (see :meth:`pnl_premium_derive`).
DERIVE_PREMIUM = _DerivePremium()


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

        The preprocessor performs seven steps:

        **Step 0b**
            ``note{...}`` / ``tags{...}`` / ``hints{...}`` bodies are lifted out
            behind an indexed placeholder, and restored in step 7. They are free
            text, so a ``#``, a ``//`` or a ``[`` in a note is prose, not DecL.
            Kept its historical name: a step 0 lifted ``doc{{{ ... }}}`` bodies
            here until 1.0.0a301, when the clause was retired
            (``dev/done/plan-decommission-docs.md``). Renumbering the rest to
            close the gap would have churned every cross reference to steps 1
            through 7 for nothing.
        **Step 1**
            Full-line comments (optional indent, then ``#`` / ``//``) are removed
            **entirely, including their newline**, so they leave no blank-line
            ghost and a comment inside a multi-line statement folds away. Done
            first so a stray bracket in a comment can never unbalance step 3.
        **Step 2**
            Trailing (inline) comments are stripped to end of line, keeping the
            line's own newline.
        **Step 3**
            Newlines inside ``[ ]`` (e.g., from formatted numpy arrays) are
            collapsed to spaces, so a vector never reads as a paragraph break.
        **Step 4**
            A ``;`` at end of line is turned into a blank-line break. Only a
            line-final ``;`` fires, so the ``;`` inside ``hints{key=value;}`` /
            ``note{...}`` (which always end a line with ``}``) is untouched.
        **Step 5**
            The text is split into paragraphs on runs of blank lines.
        **Step 6**
            Each paragraph is flattened: its newlines, indentation, and repeated
            spaces collapse to single spaces. Empty paragraphs are dropped.
        **Step 7**
            The trailer bodies lifted in step 0b are put back verbatim, so what
            reaches the lexer is exactly what was written.

        Parameters
        ----------
        program : str
            Raw multi-line DecL source.

        Returns
        -------
        list[str]
            Non-empty, whitespace-normalised DecL statements ready for parsing.
        """
        # 0b. Lift every single-line note/tags/hints body behind an indexed
        # placeholder. Same reasoning as step 0, smaller scope: the body is
        # prose, so a ``#``, ``//`` or ``[`` in it must not be read as DecL
        # punctuation by steps 2 and 3. Restored in step 7.
        bodies = []

        def _lift_trailer(m):
            bodies.append(m.group(2))
            return f'{m.group(1)}{{__DECL_TRAILER_{len(bodies) - 1}__}}'

        program = _TRAILER_BODY_RE.sub(_lift_trailer, program)

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

        # 7. Put the step-0b bodies back. Only placeholders this call wrote are
        # ever matched, so the restore cannot misfire on real note text.
        if not bodies:
            return [s for s in statements if s]
        placeholder = re.compile(r"__DECL_TRAILER_(\d+)__")
        return [placeholder.sub(lambda m: bodies[int(m.group(1))], s)
                for s in statements if s]

    @staticmethod
    def raw_statements(program: str) -> list[str]:
        """Split a program into statements, keeping each one's source layout.

        The companion to :meth:`preprocess`. Same statements, same order, but
        each is returned with its own newlines and indentation intact instead of
        flattened to one line. Backs :attr:`aggregate.recipe.Recipe.as_read`.

        Parameters
        ----------
        program : str
            Raw multi-line DecL source.

        Returns
        -------
        list[str]
            One entry per statement, in file order, comments removed and the
            statement-terminating ``;`` dropped, but otherwise as written.

        Notes
        -----
        This mirrors the *separation* rules of :meth:`preprocess` rather than
        reusing it, because ``preprocess`` rewrites the text before it splits:
        by the time it reaches its paragraph split, comment lines are gone,
        newlines inside brackets are spaces and a line-final ``;`` has become a
        blank line, so the source span of a statement no longer exists to be
        kept. Only the trailer lift is shared, and it is shared deliberately.
        ``note{...}`` bodies are free text, so a ``#``, a ``//`` or a ``;`` in
        one is prose; lifting them behind placeholders first is what stops
        ``note{layer is 5# of limit}`` losing its tail to the comment stripper
        and merging two statements into one.

        The result is **not** guaranteed equal to the matching
        :meth:`preprocess` output with its whitespace collapsed, and the
        difference is the point. ``preprocess`` reformats around brackets on its
        non-nested path, turning a source ``dfreq[1]`` into ``dfreq [1]``, and
        that path is chosen by whether the *whole file* contains a nested
        ``[[...]]``, so the same statement flattens differently depending on its
        neighbours. The contract that does hold, and the one the tests pin, is
        that each returned statement **parses to the same spec** as its
        ``preprocess`` counterpart: it is DecL, not a comment.
        """
        bodies = []

        def _lift_trailer(m):
            bodies.append(m.group(2))
            return f'{m.group(1)}{{__DECL_TRAILER_{len(bodies) - 1}__}}'

        program = _TRAILER_BODY_RE.sub(_lift_trailer, program)

        blocks, current, depth = [], [], 0
        for line in program.splitlines():
            if _FULL_LINE_COMMENT_RE.match(line):
                # transparent: never separates statements, never appears in the
                # kept text (preprocess step 1 deletes it outright)
                continue
            body = _INLINE_COMMENT_RE.sub('', line)
            if depth == 0 and not body.strip():
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append(body.rstrip())
            depth = max(0, depth + body.count('[') - body.count(']'))
            if depth == 0 and body.rstrip().endswith(';'):
                # a line-final ; terminates the statement and is not part of it
                current[-1] = current[-1].rstrip()[:-1].rstrip()
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)

        placeholder = re.compile(r'__DECL_TRAILER_(\d+)__')
        out = []
        for block in blocks:
            text = '\n'.join(block).strip('\n')
            if text.strip():
                out.append(placeholder.sub(
                    lambda m: bodies[int(m.group(1))], text))
        return out

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

    Used by ``reins_clause_part`` to distinguish a percentage share
    (``50%``) from an absolute amount (``5``) in a ``po`` reinsurance
    clause. Arithmetic on a ``_PercentNumber`` produces a plain
    ``float`` (the % marker only survives literal use), so an
    expression like ``25 * 2 %`` doesn't sneak through as a percentage.

    The marker is consumed where it is read: ``reins_clause_part``
    returns a plain ``(share, limit, attach)`` tuple, so the spec keeps
    the resolved fraction and nothing records which form was written.
    That is why :func:`aggregate.decl_writer._render_reins_clause`
    always renders the percentage form.
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

    def TAGS(self, tok):
        """Decompose ``tags{a, b c}`` into the tuple ``('a', 'b', 'c')``.

        Separators are commas and/or whitespace, so both the prose style
        (``tags{severity, heavy-tail}``) and the terse style
        (``tags{severity heavy-tail}``) work. Order is preserved and duplicates
        are dropped, so a tag list is a stable, comparable value.
        """
        body = str(tok)[5:-1]
        seen = {}
        for slug in re.split(r"[,\s]+", body):
            if slug:
                seen.setdefault(slug, None)
        return tuple(seen)

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

        _, name, kind_id, numbers, trailer = c
        return ("distortion", name,
                {**Distortion.decl_spec(kind_id, numbers), **trailer})

    def buildin_dist_list_one(self, c):
        return [c[0]]

    def buildin_dist_list_cons(self, c):
        lst, tok = c
        lst.append(tok)
        return lst

    def _resolve_combo_children(self, ids):
        """Resolve a list of ``distortion.X`` ids to actual Distortion
        instances by looking each up in the recipe base and constructing it
        from its stored spec."""
        # Local import: spectral imports nothing parser-related, but the
        # parser is imported during aggregate package init before
        # ``Distortion`` is bound at module level. Inline is safe.
        from .spectral import Distortion
        children = []
        for buildinid in ids:
            spec = self.safe_lookup(buildinid)
            children.append(Distortion.from_spec(spec))
        return children

    def distortion_out_combo(self, c):
        _, name, kind_id, child_ids, trailer = c
        if kind_id not in ('minimum', 'mixture'):
            raise ValueError(
                f"DecL: '{kind_id}' does not take a list of distortion "
                f"references; only 'minimum' and 'mixture' do")
        children = self._resolve_combo_children(child_ids)
        return ("distortion", name,
                {"name": kind_id, "distortions": children, **trailer})

    def distortion_out_combo_wtd(self, c):
        _, name, kind_id, child_ids, _wts_kw, wts, trailer = c
        if kind_id != 'mixture':
            raise ValueError(
                f"DecL: weights are only meaningful for 'mixture', not "
                f"{kind_id!r}")
        children = self._resolve_combo_children(child_ids)
        return ("distortion", name,
                {"name": kind_id, "distortions": children, "wts": wts,
                 **trailer})

    # ----- portfolio -------------------------------------------------
    def port_out(self, c):
        _, name, as_label, trailer, agg_list = c
        return ("port", name, {"spec": agg_list,
                               **as_label,
                               **trailer})

    def agg_list_cons(self, c):
        lst, ag = c
        lst.append(ag)
        return lst

    def agg_list_one(self, c):
        return [c[0]]

    # ----- aggregate -------------------------------------------------
    # ``agg_body`` is the shared aggregate body (everything after
    # ``AGG name as_label``, before the trailer), factored out so the
    # embedded engine of a ``pnl`` / ``xpnl`` reuses the identical body. Each
    # ``agg_body_*`` returns a plain spec-fragment dict (no name / as_label
    # / trailer); ``agg_out_named`` (top level) and ``agg_source_inline``
    # (embedded) add the identity and, for the top level, the trailer. See
    # dev/plan-pnl-engine-source.md.
    #
    # Until 1.0.0a231 the tweedie body also synthesised a descriptive note and
    # carried it on a private ``_engine_note`` key that both wrappers preferred
    # over the trailer, which silently destroyed the author's ``note{}``. The
    # provenance now rides the structured ``_tweedie`` key instead, the clause
    # round-trips, and no body form rewrites the trailer.
    # Interior-label temp keys emitted by the sub-object fragments (exposure /
    # FYI premium / layer / inline severity clause). They are gathered into one ``label_map``
    # sub-dict here in the body assembly and stripped from the flat spec; the
    # ``Aggregate`` (via ``LabeledMixin``) reads ``label_map``. See
    # dev/plan-labels.md ([DecL-Labels-Everywhere], S1/S2/S3).
    _INTERIOR_LABEL_KEYS = {
        "_exposure_label": "exposure",
        "_premium_label": "premium",
        "_layer_label": "layer",
        "_severity_label": "severity",
        "_wait_label": "wait",
    }

    def _pop_interior_labels(self, spec):
        """Pop the interior-label temp keys out of a merged fragment dict and
        return ``{site: label}`` for the ones that carry a (non-None) label.
        Mutates ``spec`` in place. Consumers that want labels fold the result
        into ``label_map``; consumers that don't (bivariate, clash) call it
        purely to strip the temp keys."""
        label_map = {}
        for tmp, site in self._INTERIOR_LABEL_KEYS.items():
            if tmp in spec:
                val = spec.pop(tmp)
                if val is not None:
                    label_map[site] = val
        return label_map

    def agg_body_full(self, c):
        (exposures, layers, sev_clause, occ_reins, freq, agg_reins, approx,
         orientation) = c
        spec = {
            **exposures, **layers, **sev_clause, **occ_reins, **freq,
            **agg_reins, **self._check_approx(approx, occ_reins), **orientation,
        }
        label_map = self._pop_interior_labels(spec)
        if label_map:
            spec["label_map"] = label_map
        return spec

    def agg_body_dfreq(self, c):
        (dfreq, layers, sev_clause, occ_reins, agg_reins, approx, orientation) = c
        spec = {
            **dfreq, **layers, **sev_clause, **occ_reins, **agg_reins,
            **self._check_approx(approx, occ_reins), **orientation,
        }
        label_map = self._pop_interior_labels(spec)
        if label_map:
            spec["label_map"] = label_map
        return spec

    def agg_body_renewal(self, c):
        # Sparre-Andersen renewal: ``T years`` exposure paired (strictly, at
        # the grammar level) with a ``wait``/``dwait`` clause in the freq
        # slot. ``exp_en = -1`` is the empirical sentinel (exactly like
        # ``dfreq``): the expected count is derived from the realized count
        # pmf. See dev/plan-sparre-a.md [Renewal-Frequency-Wait-Clause].
        (exposures, layers, sev_clause, occ_reins, wait_clause, agg_reins,
         approx, orientation) = c
        spec = {
            **exposures, **layers, **sev_clause, **occ_reins, **wait_clause,
            **agg_reins, **self._check_approx(approx, occ_reins), **orientation,
            "freq_name": "renewal",
            "exp_en": -1,
        }
        label_map = self._pop_interior_labels(spec)
        if label_map:
            spec["label_map"] = label_map
        return spec

    def agg_body_tweedie(self, c):
        # Tweedie distribution in reproductive (p, mean, sigma^2) form. The
        # variance function is sigma^2 * mean^p; phi = sigma^2 in Jorgenson
        # p. 127 notation. The Tweedie -> compound-Poisson(gamma)
        # reparameterization is delegated to ``tweedie_convert`` (imported
        # lazily because this module is also runnable as ``python -m`` for
        # grammar printing).
        #
        # The expansion is exact and the engine sees an ordinary poisson x
        # gamma, but ``_tweedie`` records that a ``tweedie`` clause is what was
        # written, so ``decl_writer`` renders the clause back rather than its
        # expansion. Provenance only: an aggregate written the long way carries
        # no ``_tweedie`` and still renders as its author wrote it. See
        # dev/done/plan-tweedie.md ([Tweedie-Round-Trip]).
        from .tweedie import TweedieParameters, tweedie_convert

        _tw, pp, mu, sig2 = c
        if not 1 < pp < 2:
            # The compound Poisson-gamma representation exists only strictly
            # inside (1, 2): at p = 1 the gamma shape alpha = (2-p)/(p-1) is
            # infinite and at p = 2 the Poisson rate diverges, so
            # ``tweedie_convert`` would raise a bare ZeroDivisionError. Outside
            # the interval the family has no frequency x severity form at all,
            # which is what [Power-Variance-Family] in dev/TODO.md would need
            # a different representation for.
            raise ValueError(
                f'tweedie: p must be strictly between 1 and 2, got {pp}. The '
                f'clause is `tweedie <p> <mean> <dispersion>`, shape parameter '
                f'first. (It read `<mean> <p> <dispersion>` before 1.0.0a231, '
                f'so an older program needs its first two numbers swapped.)')
        ans = tweedie_convert(p=pp, μ=mu, σ2=sig2)
        return {
            "exp_en": ans["λ"],
            "freq_name": "poisson",
            "sev_name": "gamma",
            "sev_a": ans["α"],
            "sev_scale": ans["β"],
            "_tweedie": TweedieParameters(p=pp, mean=mu, dispersion=sig2),
        }

    def agg_body_rename(self, c):
        bagg, occ_reins, agg_reins = c
        if "name" in bagg:
            del bagg["name"]
        return {**bagg, **occ_reins, **agg_reins}

    def agg_out_named(self, c):
        _, name, as_label, body, trailer = c
        return ("agg", name, {"name": name, **as_label, **body, **trailer})

    def agg_out_builtin(self, c):
        """A builtin reference, ``agg.NAME``, with its own reinsurance and trailer.

        Notes
        -----
        Only the trailer keys that carry a value are merged. ``trailer`` seeds
        ``note`` and ``hints`` with empty strings so that a freshly declared
        object gets those defaults, but ``bagg`` here is the *stored* spec of
        the referenced entry, so splatting the whole trailer overwrites the
        entry's own note and hints with nothing. That silently rebuilt every
        hinted library entry on the auto sized grid instead of the grid its
        author pinned, and made ``agg.MED.WithPicks`` fail outright, since its
        picks attachments only lie on the grid at the pinned ``bs=125``. Tags
        were already safe by accident, having no seeded default; filtering on
        truth extends the same rule to all three keys.

        An outer trailer still wins wherever it is written, because a written
        clause carries a value. The accepted limitation is that an empty
        clause on a reference, ``note{}`` were the grammar to admit it, no
        longer blanks the stored note.
        """
        bagg, agg_reins, trailer = c
        stated = {k: v for k, v in trailer.items() if v}
        return ("agg", bagg["name"], {**bagg, **agg_reins, **stated})

    # ----- profit-and-loss aggregate (premium minus loss) -----------
    def answer_pnl(self, c):
        return c[0]

    def answer_xpnl(self, c):
        return c[0]

    def _attach_pnl(self, spec, premium):
        """Record the consideration (the stated ``pnl`` premium) on a spec.

        The premium is a single deterministic amount for the book (an
        aggregate-level affine shift), in contrast to a constant inside
        ``sev``/``dsev``/``ssev`` which is per-claim. ``inherit premium`` defers
        resolution to the factory (the sentinel :data:`INHERIT_PREMIUM`), which
        reads the built engine's technical premium.

        Parameters
        ----------
        spec : dict
            The pnl spec (mutated in place).
        premium : float, list, INHERIT_PREMIUM, or DERIVE_PREMIUM
            The stated premium, recorded as ``consideration``.
        """
        spec["consideration"] = premium

    # ----- the wrapped engine (agg_source) ---------------------------
    # A ``pnl`` / ``xpnl`` wraps a complete stochastic engine and reads its loss
    # out. Each ``agg_source_*`` returns a ``(kind, name, spec)`` triple; kind is
    # ``'agg'`` for an inline body or an ``agg.NAME`` reference (both merge into
    # the pnl spec so the existing plain / GCN / retro / var / reinstatement
    # factory path is reused unchanged), or ``'port'`` for a ``port.NAME``
    # reference (the factory builds a Portfolio and reads its net-net total).
    def agg_source_inline(self, c):
        # ``agg NAME <body>`` -- a complete inline aggregate, no trailer (the
        # wrapping pnl/xpnl owns it). Returns the engine spec fragment; the
        # engine's display label is carried through so the inner Aggregate is
        # faithfully the declared engine, and so is a ``_tweedie`` provenance
        # key, which is what lets a tweedie engine inside a pnl render its
        # clause back.
        _agg, name, as_label, body = c
        return ("agg", name, {"name": name, **as_label, **body})

    def agg_source_ref_agg(self, c):
        # ``agg.NAME`` (optionally scaled) -- ``builtin_agg`` already resolves it
        # to a deep-copied spec dict via safe_lookup.
        bagg = c[0]
        return ("agg", bagg.get("name"), bagg)

    def agg_source_inline_port(self, c):
        # ``port PNAME <units>`` -- a complete inline portfolio, no trailer (the
        # wrapping pnl/xpnl owns it), the twin of ``agg_source_inline``. Shaped
        # exactly like ``port_out``'s spec so the two are interchangeable
        # downstream: the underwriter builds the engine from this dict, and the
        # writer renders it back as the same nested block.
        _port, name, as_label, agg_list = c
        return ("port", name, {"name": name, "spec": agg_list, **as_label})

    def agg_source_ref_port(self, c):
        # ``port.NAME`` -- resolve the stored portfolio spec; the pnl reads the
        # net-net total density. The factory builds the Portfolio engine.
        portid = str(c[0])
        portname = portid.split(".", 1)[1]
        portspec = self.safe_lookup(portid)
        # ``port.ref`` rather than ``port``: the spec is the same shape either
        # way, and the kind is what tells the writer to render the reference
        # back as a reference instead of expanding the units it resolved to.
        return ("port.ref", portname, {**portspec, "name": portname})

    def _pnl_spec(self, kind, name, as_label, premium, source, expense,
                  peel, trailer):
        """Assemble the ``(kind, name, spec)`` tuple shared by ``pnl``/``xpnl``.

        For an inline / ``agg.NAME`` engine the loss structure is merged into the
        pnl spec (the pnl owns identity; the engine's name/note/hints/label are
        set aside), so the underwriter's existing pnl factory path is reused. A
        portfolio engine cannot merge into a loss-agg spec, so its whole spec is
        recorded under ``_engine_port_spec`` for a dedicated factory path.

        **Both portfolio forms carry the spec**, whether the source wrote the
        units out (``port PNAME <units>``) or referenced them (``port.NAME``,
        which ``agg_source_ref_port`` resolves here at parse time). The
        underwriter therefore never looks a portfolio up, and the two forms
        differ only in how the writer renders them back: ``_engine_port`` holds
        the referenced *name* and is set for the reference form alone, so it
        round-trips as ``port.NAME`` rather than being expanded.
        """
        ekind, ename, espec = source
        spec = {"name": name, **as_label, **expense, **peel, **trailer}
        if ekind in ("port", "port.ref"):
            spec["_engine_port_spec"] = espec
            if ekind == "port.ref":
                spec["_engine_port"] = ename
        else:
            for k, v in espec.items():
                # ``tags`` joins the existing skip list: the pnl owns its
                # own metadata, and an engine's tags are not the pnl's tags.
                if k in ("name", "note", "hints", "label", "tags"):
                    continue
                spec[k] = v
            # The engine's own note and label are inner-Aggregate presentation;
            # keep them without shadowing the pnl's trailer/label. (A tweedie
            # engine's ``_tweedie`` is not metadata and is not skipped above, so
            # it rides the loop into the pnl spec and the clause round-trips.)
            if espec.get("note"):
                spec["engine_note"] = espec["note"]
            if "label" in espec:
                spec["engine_label"] = espec["label"]
            # The engine's own name is cosmetic (discarded at build) but is
            # preserved so it round-trips through the writer instead of being
            # re-synthesized as ``NAME_e``.
            if ename:
                spec["engine_name"] = ename
        self._attach_pnl_head(spec, premium)
        return (kind, name, spec)

    def pnl_out_engine(self, c):
        (_pnl, name, as_label, premium, _less, source, expense, peel,
         trailer) = c
        return self._pnl_spec("pnl", name, as_label, premium, source,
                              expense, peel, trailer)

    def xpnl_out_engine(self, c):
        (_xpnl, name, as_label, premium, _less, source, expense, peel,
         trailer) = c
        return self._pnl_spec("xpnl", name, as_label, premium, source,
                              expense, peel, trailer)

    # ----- the layer-peeling clause ----------------------------------
    #: Accepted ``peel`` directions ([Layer-Peeling-Shorthand]). ``top-down``
    #: introduces the highest-attaching layer first, ``bottom-up`` the lowest.
    _PEEL_DIRECTIONS = ("top-down", "bottom-up")

    def peel_set(self, c):
        """``peel DIRECTION`` -> ``{'peel': DIRECTION}`` (direction validated)."""
        _peel, direction = c
        direction = str(direction)
        if direction not in self._PEEL_DIRECTIONS:
            raise ValueError(
                f"DecL: peel '{direction}' is not recognised; "
                f"use one of {', '.join(self._PEEL_DIRECTIONS)}")
        return {"peel": direction}

    def peel_none(self, _c):
        """No ``peel`` clause: add no key, so the tier walk is the default.

        Returning an empty dict (rather than an explicit default) keeps every
        program that does not peel byte-identical through the writer and leaves
        the frozen spec snapshots untouched.
        """
        return {}

    # ----- gross-premium head: fixed amount or retro rating clause ---
    def pnl_premium_fixed(self, c):
        """``<num> premium [as <label>]`` -- the fixed gross premium.

        Returns a dict carrying the amount (``_premium``) and an optional
        consideration display label (``_label``), consumed by
        :meth:`_attach_pnl_head`.
        """
        numbers, _prem, as_label = c
        head = {'_premium': numbers}
        if 'label' in as_label:
            head['_label'] = as_label['label']
        return head

    def pnl_premium_inherit(self, c):
        """``inherit premium [as <label>]`` -- copy the engine's technical premium.

        Resolution is deferred to the factory (the engine is not built at parse
        time): the head carries the :data:`INHERIT_PREMIUM` sentinel, and the
        underwriter reads the built engine's ``exp_premium`` (an :class:`Aggregate`)
        or the accumulated portfolio premium, erroring if the engine has none.
        """
        _inherit, _prem, as_label = c
        head = {'_premium': INHERIT_PREMIUM}
        if 'label' in as_label:
            head['_label'] = as_label['label']
        return head

    def pnl_premium_derive(self, c):
        """``derive premium [as <label>]``: the engine premium grossed up for expenses.

        Resolution is deferred to the factory exactly as ``inherit premium``:
        the head carries the :data:`DERIVE_PREMIUM` sentinel, and the
        underwriter reads the engine's technical premium T and the ``less``
        clause, giving ``(T + fixed total) / (1 - premium ratio total)``.
        Fixed and premium expense bases only; a loss basis expense, an engine
        without premium, and premium ratios totalling one or more are build
        errors.
        """
        _derive, _prem, as_label = c
        head = {'_premium': DERIVE_PREMIUM}
        if 'label' in as_label:
            head['_label'] = as_label['label']
        return head

    def pnl_premium_retro(self, c):
        """``retro <collar> premium [as <label>]`` -- the account-level
        retrospective rating clause (Phase 3); returns the collar tagged for
        :meth:`_attach_pnl_head`, plus an optional consideration label."""
        _retro, collar, _prem, as_label = c
        head = {'_retro': collar}
        if 'label' in as_label:
            head['_label'] = as_label['label']
        return head

    def _attach_pnl_head(self, spec, head):
        """Record the gross-premium head: a fixed consideration, or a retro clause.

        A fixed head carries ``_premium`` (a number), recorded as
        ``consideration`` (delegating to :meth:`_attach_pnl`). A retro head carries
        the keyword-first collar dict under ``_retro``; it is recorded as the
        account-level ``retro_terms`` spec key and uses the collar ``basic`` as the
        representative consideration (the actual gross premium is the variable map,
        resolved by the variable-rating builder). Either may carry an optional
        ``_label`` -- the consideration leg's display label, recorded as
        the ``consideration_label`` spec key.
        """
        if '_retro' in head:
            collar = head['_retro']
            spec['retro_terms'] = collar
            self._attach_pnl(spec, collar['basic'])
        else:
            self._attach_pnl(spec, head['_premium'])
        if head.get('_label') is not None:
            spec['consideration_label'] = head['_label']

    # ----- gross expenses on a pnl (two-level: groups of terms) ------
    # Each term is a ``(basis, value)`` pair. ``and``-joined terms collect into
    # one **group** (``expense_terms``); a group carries an optional ``as`` label
    # (``expense_group``); juxtaposed groups (``expense_groups``) stay separate.
    # ``expense_some`` records the whole thing as ``expense_spec`` -- a list of
    # ``(label, [(basis, value), ...])`` groups, one obligation leg per group.
    # See dev/plan-decl-labels.md.
    def expense_premium(self, c):
        # ``<frac> premium expenses``: variable expense, base = gross premium.
        return ("premium", float(c[0]))

    def expense_loss(self, c):
        # ``<frac> loss expenses``: variable expense, base = expected gross loss.
        return ("loss", float(c[0]))

    def expense_fixed(self, c):
        # ``<amount> fixed expenses``: a fixed currency amount.
        return ("fixed", float(c[0]))

    def expense_terms_one(self, c):
        return [c[0]]

    def expense_terms_cons(self, c):
        lst, _and, term = c
        lst.append(term)
        return lst

    def expense_group(self, c):
        # c = [terms_list, as_label] -> (label_or_None, terms_list)
        terms, as_label = c
        return (as_label.get("label"), terms)

    def expense_groups_one(self, c):
        return [c[0]]

    def expense_groups_cons(self, c):
        lst, group = c
        lst.append(group)
        return lst

    def expense_less_some(self, c):
        # ``less <expense-groups>`` -- the second-``less`` clause.
        _less, groups = c
        return {"expense_spec": groups}

    def expense_less_none(self, c):
        return {}

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
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
        return ("bvagg", name, spec)

    def bv_out_copula_nofreq(self, c):
        _mv, name, exposures, body, copula, trailer = c
        spec = {
            "name": name,
            **exposures,
            "freq_name": "poisson",
            "units": body,
            "copula": copula,
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
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
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
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
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
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
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
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
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
        return ("bvagg", name, spec)

    @staticmethod
    def _bv_out_viewpair(c, views):
        """Shared builder for the three occurrence view-pair prefixes.

        ``<keyword> <agg with occurrence reinsurance>`` -> the joint
        per-occurrence aggregate of the named pair of {gross, ceded, net} as a
        ``netceded``-mode BivariateAggregate. ``views`` is the ``(x, y)`` axis
        pair the keyword names (e.g. ``('net', 'ceded')`` for ``netceded``).

        The grammar here is ``NETCEDED agg_out``: the statement has no trailer
        of its own, so a ``note`` / ``tags`` / ``hints`` / ``doc`` written on
        the line lands on the inner ``agg`` and is lifted onto the view-pair.
        The inner agg keeps its copy -- there is only one statement, so the two
        describe the same thing.
        """
        _kw, agg_tuple = c          # agg_tuple = ("agg", name, spec)
        _, name, spec = agg_tuple
        out = {
            "name": name,
            "mode": "netceded",
            "nc_views": views,
            "units": [agg_tuple],
            "note": spec.get("note", ""),
            "hints": spec.get("hints", ""),
        }
        # tags is a conditional key: lift only when present.
        for key in ("tags",):
            if spec.get(key):
                out[key] = spec[key]
        return ("bvagg", name, out)

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
        spec = {**layers, **sev_clause}
        # Clash components are out of scope for labels; strip any interior-label
        # temp keys so they don't leak into the clash spec (dev/plan-labels.md).
        self._pop_interior_labels(spec)
        return spec

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
            **trailer,
        }
        # Bivariate reuses the shared ``exposures`` production but is out of
        # scope for labels (dev/plan-labels.md); strip any interior-label temp
        # keys so they don't leak into the bivariate spec.
        self._pop_interior_labels(spec)
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
        _, name, as_label, sev, trailer = c
        sev["name"] = name
        sev.update(as_label)
        sev.update(trailer)
        return ("sev", name, sev)

    def sev_out_dsev(self, c):
        _, name, as_label, dsev, trailer = c
        dsev["name"] = name
        dsev.update(as_label)
        dsev.update(trailer)
        return ("sev", name, dsev)

    # ----- frequency -------------------------------------------------
    # Zero modification / truncation. The bare forms leave ``freq_pin_mean``
    # unset, which is now the *pinned* reading: ``Aggregate`` defaults it True
    # and solves for the base mean whose realized E[N] equals the exposure
    # clause, so ``10 claims ... zt`` delivers ten claims. The ``!`` variants
    # opt out and set it False, giving the textbook (a, b, 1) base
    # parameterization where the clause sets the base mean and the reweighting
    # shifts the realized E[N] off it ([ZT-ZM-Recalibrate-Default]).
    #
    # The rule names still read from the grammar, where the ``!`` alternatives
    # are the ``_pin`` ones. Renaming them would be a grammar edit for no gain:
    # the marker is what moved, not the shape of the language.
    def freq_zm(self, c):
        freq, _zm, expr = c
        freq["freq_zm"] = True
        freq["freq_p0"] = expr
        return freq

    def freq_zm_pin(self, c):
        freq, _zm, expr = c
        freq["freq_zm"] = True
        freq["freq_p0"] = expr
        freq["freq_pin_mean"] = False
        return freq

    def freq_zt(self, c):
        freq, _zt = c
        freq["freq_zm"] = True
        freq["freq_p0"] = 0.0
        return freq

    def freq_zt_pin(self, c):
        freq, _zt = c
        freq["freq_zm"] = True
        freq["freq_p0"] = 0.0
        freq["freq_pin_mean"] = False
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
    @staticmethod
    def _split_reins(triples, which, kind):
        """Split a reins_list of ``(layer, premium, cede, reinst)`` into spec keys.

        ``<which>_reins`` is the list of ``(share, limit, attach)`` layer tuples
        the reinsurance engine consumes (unchanged). The parallel per-layer
        ``<which>_reins_premium`` (``(basis, value)`` or ``None``),
        ``<which>_reins_cede`` (fraction or ``None``) and ``<which>_reins_reinst``
        (reinstatement-rate tuple or ``None``) lists are added **only** when some
        layer carries them, so plain reinsurance specs are untouched.

        The cross-layer reinstatement constraints (locked, dev/plan-reinstatements.md)
        are enforced here, where the whole layer list is in scope: reinstatements
        decorate an **occurrence** layer only; **at most one** layer may carry the
        clause ([reins-one-clause]); and the occurrence tier must then be a
        **single** layer ([reins-single-layer], the 2-D ``(L, R)`` engine ceiling).
        """
        layers = [t[0] for t in triples]
        prem = [t[1] for t in triples]
        cede = [t[2] for t in triples]
        reinst = [t[3] for t in triples]
        variable = [t[4] for t in triples]
        label = [t[5] for t in triples]
        out = {f"{which}_reins": layers, f"{which}_kind": kind}
        if any(p is not None for p in prem):
            out[f"{which}_reins_premium"] = prem
        if any(x is not None for x in cede):
            out[f"{which}_reins_cede"] = cede
        # per-cession display labels (the ``as`` clause) -- emitted only when some
        # layer carries one, so plain reinsurance specs are untouched. Consumed by
        # the underwriter to name the cession group / ledger rows.
        if any(x is not None for x in label):
            out[f"{which}_reins_label"] = label
        # variable rating (Phase 3): one of swing / slide / pc / corridor on one
        # aggregate layer (decision 0: at most one feature per program, no
        # stacking; occurrence-basis variable rating is a follow-up). Emit the
        # locked per-feature spec key ``<which>_reins_<feature>`` and record the
        # decorated layer index so the underwriter resolves its economics.
        present = [(i, v) for i, v in enumerate(variable) if v is not None]
        if present:
            if which != "agg":
                raise ValueError(
                    "DecL: variable-rating features (swing / slide / pc / "
                    "corridor) are supported on aggregate reinsurance only in "
                    "this release; occurrence-basis variable rating is a "
                    "follow-up. (Reinstatements remain the occurrence "
                    "stochastic-ceded feature.)")
            if len(present) > 1:
                raise ValueError(
                    f"DecL: at most one variable-rating feature per program "
                    f"(decision 0: no stacking); found {len(present)}.")
            idx, (feat, params) = present[0]
            out[f"{which}_reins_{feat}"] = params
            out[f"{which}_reins_{feat}_layer"] = idx
        n_reinst = sum(1 for r in reinst if r is not None)
        if n_reinst:
            if which != "occ":
                raise ValueError(
                    "DecL: a 'reinstatements' clause decorates an occurrence "
                    "layer, not aggregate reinsurance.")
            if n_reinst > 1:
                raise ValueError(
                    f"DecL: 'reinstatements' may decorate at most one occurrence "
                    f"layer; found {n_reinst} ([reins-one-clause]).")
            if len(layers) != 1:
                raise ValueError(
                    f"DecL: 'reinstatements' require a single occurrence layer; "
                    f"found {len(layers)} ([reins-single-layer]). A reinstated "
                    "layer cannot share the occurrence tier with other layers "
                    "(the engine ceiling is the 2-D (L, R) joint).")
            out[f"{which}_reins_reinst"] = reinst
        return out

    def agg_reins_net(self, c):
        return self._split_reins(c[3], "agg", "net of")

    def agg_reins_ceded(self, c):
        return self._split_reins(c[3], "agg", "ceded to")

    def agg_reins_none(self, c):
        return {}

    def occ_reins_net(self, c):
        return self._split_reins(c[3], "occ", "net of")

    def occ_reins_ceded(self, c):
        return self._split_reins(c[3], "occ", "ceded to")

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

    # ----- orientation suffix (value_type) ---------------------------
    def orientation_payoff(self, c):
        """``payoff`` suffix -> ``{'value_type': 'payoff'}``.

        Pure orientation: sets the sign-convention role only. No reflect /
        shift (that affine is the ``pnl`` wrapper); pricing reads
        ``_is_loss_value`` and applies the dual distortion. See
        dev/plan-pnl.md S4.
        """
        return {"value_type": "payoff"}

    def orientation_loss(self, c):
        """``loss`` suffix -> ``{'value_type': 'loss'}`` (explicit default)."""
        return {"value_type": "loss"}

    def orientation_none(self, c):
        """Omitted orientation -> no spec key (constructor default ``loss``)."""
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
        # A tower has no per-layer premium / cede / reinstatements / variable
        # feature / label: wrap each layer as a ``(layer, premium, cede, reinst,
        # variable, label)`` so reins_list elements are uniform.
        tower = c[0]
        limit, attach = tower[0], tower[1]
        return [((1.0, l, a), None, None, None, None, None)
                for l, a in zip(limit, attach)]

    def reins_clause_xs(self, c):
        limit, _xs, attach = c
        return (1.0, limit, attach)

    def reins_clause_part(self, c):
        # ``po`` (part of) is the one partial-placement keyword. The leading
        # quantity says which reading is meant: a literal percentage (``50%``)
        # is the share directly, a bare number is an absolute amount and the
        # share is ``amount / limit``. The ``_PercentNumber`` carries the
        # ``%``-suffix marker through the parse so this branch can decide.
        # ``so`` / ``of`` were retired at 1.0.0a249 (identical arithmetic).
        n, _po, limit, _xs, attach = c
        if isinstance(n, _PercentNumber):
            return (float(n), limit, attach)
        if n / limit < 0.05:
            logger.warning(
                f"'{n:.6g} po {limit:.6g} xs {attach:.6g}' reads {n:.6g} as an "
                f"amount, giving a {n / limit:.4%} placement, which is "
                f"suspiciously small. If you meant a share, write it as a "
                f"percentage: '{n * 100:.6g}% po {limit:.6g} xs {attach:.6g}'."
            )
        return (n / limit, limit, attach)

    # ----- ceded-premium / commission decorators (decision 3) -------
    def reins_clause(self, c):
        """Combine a loss layer with its optional premium / cede / reinstatements
        / variable-rating feature.

        Returns ``(layer, premium, cede, reinst, variable, label)`` where ``layer``
        is the ``(share, limit, attach)`` tuple consumed by the reinsurance engine,
        ``premium`` is ``(basis, value)`` (basis in ``deposit`` / ``rol`` /
        ``rate``) or ``None``, ``cede`` is the commission fraction or ``None``,
        ``reinst`` is the tuple of reinstatement price multipliers or ``None``,
        ``variable`` is ``(feature, params)`` for one of swing / slide / pc /
        corridor or ``None``, and ``label`` is the optional ``as`` display label
        for the cession (or ``None``). ``reins_list`` splits these into parallel
        spec keys.
        """
        layer, premium, cede, reinst, variable, as_label = c
        label = as_label.get("label")
        if cede is not None and premium is None:
            raise ValueError(
                "DecL: 'cede' (ceding commission) needs a ceded-premium clause "
                "(deposit / rol / rate) on the same layer.")
        if premium is not None and premium[0] == 'rol' and not np.isfinite(layer[1]):
            raise ValueError(
                "DecL: 'rol' (rate on line) needs a finite limit "
                "(rate-on-line is a fraction of share x limit).")
        if reinst is not None and premium is None:
            raise ValueError(
                "DecL: a 'reinstatements' clause needs a base premium clause "
                "(deposit / rol / rate) on the same layer; the base rate on "
                "line r = base_premium / limit would otherwise be undefined "
                "([reins-premium]).")
        if variable is not None:
            feat = variable[0]
            if reinst is not None:
                raise ValueError(
                    f"DecL: '{feat}' cannot combine with 'reinstatements' "
                    "(decision 0: one variable-rating feature per program).")
            if feat == 'swing':
                if premium is not None:
                    raise ValueError(
                        "DecL: 'swing' supplies the ceded premium and replaces "
                        "the deposit / rol / rate clause; give one or the other.")
            else:
                # slide / pc / corridor read the ceded loss ratio, so they need a
                # fixed ceded-premium denominator.
                if premium is None:
                    raise ValueError(
                        f"DecL: '{feat}' reads the ceded loss ratio and needs a "
                        "ceded-premium clause (deposit / rol / rate) for the "
                        "denominator.")
                if feat == 'slide' and cede is not None:
                    raise ValueError(
                        "DecL: 'slide' replaces the fixed 'cede' commission; give "
                        "one or the other.")
        return (layer, premium, cede, reinst, variable, label)

    def reins_premium_deposit(self, c):
        return ('deposit', float(c[1]))

    def reins_premium_rol(self, c):
        return ('rol', float(c[1]))

    def reins_premium_rate(self, c):
        return ('rate', float(c[1]))

    def reins_premium_none(self, c):
        return None

    def reins_cede_some(self, c):
        return float(c[1])

    def reins_cede_none(self, c):
        return None

    # ----- variable-rating feature decorator (Phase 3) --------------
    # One of swing / slide / pc / corridor optionally decorates a layer; each
    # returns ``(feature, params)`` carried on the spec key
    # ``<which>_reins_<feature>`` and consumed by the underwriter to build the
    # matching ContractTerms. See dev/plan-variable-rating.md.
    def reins_var_none(self, c):
        return None

    def reins_var_swing(self, c):
        _kw, collar = c
        return ('swing', collar)

    def reins_var_slide(self, c):
        _kw, anchors = c
        return ('slide', {'anchors': tuple(anchors)})

    def reins_var_pc(self, c):
        _kw, share, _after, allowance = c
        return ('pc', {'share': float(share), 'allowance': float(allowance)})

    def reins_var_corridor(self, c):
        _kw, share, _po, width, _xs, attach = c
        return ('corridor', {'share': float(share), 'width': float(width),
                             'attachment': float(attach)})

    def collar(self, c):
        """``basic <b> lcm <m> [min <lo>] [max <hi>]`` -> the collar dict."""
        _basic, basic, _lcm, lcm, minimum, maximum = c
        return {'basic': float(basic), 'lcm': float(lcm),
                'minimum': minimum, 'maximum': maximum}

    def collar_min_some(self, c):
        return float(c[1])

    def collar_min_none(self, c):
        return None

    def collar_max_some(self, c):
        return float(c[1])

    def collar_max_none(self, c):
        return None

    def slide_anchor(self, c):
        comm, _at, lr = c
        return (float(comm), float(lr))

    def slide_anchors_one(self, c):
        return [c[0]]

    def slide_anchors_cons(self, c):
        lst, _and, anchor = c
        lst.append(anchor)
        return lst

    # ----- reinstatement schedule decorator (property-cat) ----------
    # Both surface forms (explicit ``[alpha ...]`` list and the ``<count> free /
    # <count> at <p>%`` group chain) reduce to a flat tuple of price multipliers
    # alpha_j, carried on the spec key ``occ_reins_reinst`` and consumed by the
    # underwriter to build a ReinstatementTerms. ``free`` = ``at 0%``.
    _NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}

    def reins_reinst_none(self, c):
        return None

    def reins_reinst_zero(self, c):
        """``no reinstatements`` -> zero reinstatements (a single annual limit).

        The empty tuple is the distinct marker (vs ``None`` = omitted = free +
        unlimited): ``m = 0``, recovery capped at the single occurrence limit
        ``y``, no reinstatement premium.
        """
        return ()

    def reins_reinst_list(self, c):
        """``reinstatements [a1 a2 ...]`` -- the explicit price-multiplier list."""
        _kw, rates = c
        return self._reinst_rates(rates)

    def reins_reinst_groups(self, c):
        """``reinstatements <count> free and <count> at <p>% and ...``."""
        _kw, groups = c
        return self._reinst_rates(groups)

    @staticmethod
    def _reinst_rates(rates):
        """Validate and freeze a reinstatement price-multiplier sequence."""
        out = tuple(float(a) for a in rates)
        if len(out) < 1:
            raise ValueError(
                "DecL: a 'reinstatements' clause needs at least one rate.")
        if any((not np.isfinite(a)) or a < 0 for a in out):
            raise ValueError(
                "DecL: reinstatement rates must be finite and nonnegative, "
                f"got {out!r}.")
        return out

    def reinst_groups_one(self, c):
        return list(c[0])

    def reinst_groups_cons(self, c):
        lst, _and, grp = c
        lst.extend(grp)
        return lst

    def reinst_group_free(self, c):
        # ``<count> free`` -> ``count`` free (zero-rate) reinstatements.
        count = c[0]
        return [0.0] * count

    def reinst_group_at(self, c):
        # ``<count> at <p>%`` -> ``count`` reinstatements at multiplier ``p``.
        count, _at, mult = c
        return [float(mult)] * count

    def count_number(self, c):
        n = c[0]
        if not (float(n) > 0 and float(n) == int(n)):
            raise ValueError(
                f"DecL: a reinstatement count must be a positive integer, "
                f"got {n!r}.")
        return int(n)

    def count_word(self, c):
        word = str(c[0]).lower()
        if word not in self._NUMBER_WORDS:
            raise ValueError(
                f"DecL: '{c[0]}' is not a valid reinstatement count; use a "
                f"digit or one of {', '.join(self._NUMBER_WORDS)}.")
        return self._NUMBER_WORDS[word]

    # ----- severity (continuous) ------------------------------------
    def sev_clause_sev(self, c):
        # A reflected severity (``-X`` / ``shift - X``) is accepted here and
        # built like any other severity whose support reaches below zero
        # (``10 * norm + 5``, ``lognorm 5 cv 1 - 10``): the layered-loss
        # transform clamps x<0 -> 0, the severity stays non-signed, and layers
        # and occurrence reinsurance work normally. When the reflected support
        # does reach below zero the Severity warns, names the clamped mass, and
        # points at ``ssev``. See dev/done/plan-reflected-loss-severity.md; the
        # rejection this replaced is in dev/done/plan-decl-sev-unary-minus.md.
        _sev, sev, as_label = c
        sev["_severity_label"] = as_label.get("label")
        return sev

    def sev_clause_ssev(self, c):
        # ssev = signed (never-clamp) continuous severity: a profit is a
        # negative loss. Same spec as sev, flagged so the Severity keeps its
        # negative support instead of clamping x<0 -> 0. Orthogonal to
        # value_type (does NOT imply payoff). See dev/plan-negative-x-agg.md.
        _ssev, sev, as_label = c
        sev['sev_signed'] = True
        sev["_severity_label"] = as_label.get("label")
        return sev

    def sev_clause_dsev(self, c):
        dsev, as_label = c
        dsev["_severity_label"] = as_label.get("label")
        return dsev

    def sev_clause_builtin(self, c):
        builtin, as_label = c
        b = self.safe_lookup(builtin)
        if "name" in b:
            del b["name"]
        b["_severity_label"] = as_label.get("label")
        return b

    # ----- severity reference (agg.NAME / port.NAME) ------------------
    # DecL's first DEFERRED reference. Every other dotted reference is
    # resolved and inlined right here, by ``safe_lookup``; this one cannot be,
    # because the severity is the inner's *computed output* (``xs`` /
    # ``agg_density``), which exists only after an update. So the parse records
    # a symbolic ``sev_ref`` and ``Underwriter._resolve_sev_ref`` resolves it at
    # build time. The precedent for a symbolic marker the unparser renders back
    # is ``_engine_port_spec`` / ``_engine_port`` on a ``pnl``.
    #
    # The existence check still happens now, so an unknown name fails at parse
    # time where the user can see it; the spec copy it returns is discarded.

    def _sev_ref_spec(self, token):
        """Symbolic spec for one ``agg.NAME`` / ``port.NAME`` severity reference."""
        ref = str(token)
        # kind-checked existence check; the returned spec copy is deliberately
        # thrown away (resolution is deferred to build time)
        self.safe_lookup(ref)
        return {"sev_ref": ref}

    def sev_ref_agg(self, c):
        return self._sev_ref_spec(c[0])

    def sev_ref_port(self, c):
        return self._sev_ref_spec(c[0])

    def sev_ref_uncond(self, c):
        # the SEVERITY-side ``!``: keep the layer unconditional, so a zero atom
        # in the referenced law survives an outer layers clause. Nothing to do
        # with the frequency-side ``!`` on zt / zm.
        ref = c[0]
        ref["sev_conditional"] = False
        return ref

    def sev_clause_ref(self, c):
        _sev, ref, as_label = c
        ref["_severity_label"] = as_label.get("label")
        return ref

    def sev_clause_ref_signed(self, c):
        _ssev, ref, as_label = c
        ref["sev_signed"] = True
        ref["_severity_label"] = as_label.get("label")
        return ref

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

    def sev1_negate(self, c):
        """``- X``: bare unary minus on a severity -> reflect it (``0 - X``).

        Sugar for the working ``0 - X`` (``sev2_rsub`` with an implicit zero
        shift). Unary minus binds *tighter* than the additive ``+/- c`` shift
        (standard math precedence), so it negates at the ``sev1`` level and a
        trailing shift wraps the reflected term: ``-lognorm 2 + 5`` parses as
        ``(-X) + 5 == 5 - X``, *not* ``-(X + 5)``.

        ``X`` (``sev1``) has value ``Lx + s*base`` with ``s = -1`` if already
        reflected else ``+1``. A pure reflection ``-X = -Lx + (-s)*base`` negates
        any location and toggles ``sev_reflect``. An *absent* ``sev_loc`` is left
        absent (``-0 == 0``), matching the ``sev1_scaled`` convention so the form
        is byte-identical to the canonical ``-1 * X`` under the unparser. Legal
        under both keywords: ``ssev -X`` keeps the negative support, ``sev -X``
        clamps it, which for a pure negation means the whole law clamps and the
        severity is a point mass at 0 (with a warning). See
        dev/done/plan-reflected-loss-severity.md.
        """
        _minus, sev1 = c
        if "sev_loc" in sev1:
            sev1["sev_loc"] = -_check_vectorizable(sev1["sev_loc"])
        sev1["sev_reflect"] = not sev1.get("sev_reflect", False)
        return sev1

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

    # ----- wait / dwait (renewal waiting-time clause) ----------------
    @staticmethod
    def _sev_to_wait(d):
        """Rename a parsed severity fragment into the flat ``wait_*`` keys.

        The ``wait`` clause reuses the full severity mini-language, so the
        sub-parse arrives as ``sev_*`` keys; the wait law stores them as the
        mirrored ``wait_*`` family. A ``sev.NAME`` lookup's identity baggage
        (``name``/``note``/``hints``/``label``/``label_map``) is dropped --
        the wait borrows the distribution, not the object. Picks and
        signed/reflected forms are rejected: neither is meaningful for a
        non-negative waiting time.
        """
        if any(k.startswith("sev_pick") and d[k] is not None for k in d):
            raise ValueError(
                "DecL: picks are not meaningful on a wait clause")
        if d.get("sev_signed", False) or d.get("sev_reflect", False):
            raise ValueError(
                "DecL: a wait clause cannot be signed or reflected "
                "(waiting times are non-negative)")
        drop = {"name", "note", "hints", "label", "label_map",
                "sev_signed", "sev_reflect"}
        out = {}
        for k, v in d.items():
            if k in drop or k.startswith("sev_pick"):
                continue
            out["wait_" + k[4:] if k.startswith("sev_") else k] = v
        return out

    def wait_clause_wait(self, c):
        _wait, sev, as_label = c
        spec = self._sev_to_wait(sev)
        spec["_wait_label"] = as_label.get("label")
        return spec

    def wait_clause_layer(self, c):
        # ``wait y xs a <dist>``: the severity layer transform on the wait
        # law. Unpack order matches ``layers_xs`` (limit xs attachment).
        # Splice + layer on the same wait is rejected -- the defective-window
        # (splice ``!``) and layer (atom at 0 / cap at y) semantics of ``!``
        # would collide on one flag.
        _wait, limit, _xs, attach, sev, as_label = c
        spec = self._sev_to_wait(sev)
        if (np.any(np.atleast_1d(spec.get("wait_lb", 0)) != 0)
                or np.any(np.atleast_1d(spec.get("wait_ub", np.inf)) != np.inf)):
            raise ValueError(
                "DecL: a wait clause cannot combine a splice window "
                "[lb ub] with a layer (y xs a); use one or the other")
        spec["wait_limit"] = limit
        spec["wait_attachment"] = attach
        spec["_wait_label"] = as_label.get("label")
        return spec

    def wait_clause_dwait(self, c):
        # prob-sum policy (checked here, once the optional trailing ``!`` is
        # known): conditional renormalizes with a warning when off by more
        # than 1e-6; defective (``!``) accepts sum <= 1, errors above.
        dwait, as_label = c
        ps = np.asarray(dwait["wait_ps"], dtype=float)
        s = float(ps.sum())
        if dwait.get("wait_conditional", True):
            if abs(s - 1.0) > 1e-6:
                logger.warning(
                    'dwait probabilities sum to %.8g != 1; renormalizing', s)
                dwait["wait_ps"] = ps / s
        elif s > 1.0 + 1e-12:
            raise ValueError(
                f'defective dwait probabilities sum to {s:.8g} > 1')
        dwait["_wait_label"] = as_label.get("label")
        return dwait

    def dwait_main(self, c):
        _dwait, doutcomes, dprobs = c
        ps = np.ones_like(doutcomes) / len(doutcomes) if len(dprobs) == 0 else dprobs
        return {"wait_name": "dhistogram", "wait_xs": doutcomes,
                "wait_ps": ps}

    def dwait_unconditional(self, c):
        dwait = c[0]
        dwait["wait_conditional"] = False
        return dwait

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
        limit, _xs, attach, as_label = c
        return {"exp_attachment": attach, "exp_limit": limit,
                "_layer_label": as_label.get("label")}

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

    # ----- trailer (optional note / tags / hints) ---------------------
    # Each item method returns a ``(key, value)`` pair; ``trailer`` folds them
    # into one dict that call sites splat with ``**trailer``.
    #
    # ``note`` and ``hints`` are ALWAYS present (possibly empty) because every
    # spec has carried them since 1.0.0a25 and the captured spec snapshot
    # compares key sets exactly (tests/test_decl_parser.py). ``tags`` is added
    # ONLY when written -- adding it unconditionally would put
    # ``Extra={'tags'}`` into every spec and fail all 163 snapshot cases.
    # Hosts therefore read it with ``spec.get(...)``.
    #
    # ``hints`` is left as a raw ``key=value;`` string here; it is parsed and
    # type-coerced in ``aggregate.underwriter`` (caller-wins merge).
    def trailer_item_note(self, c):
        return ("note", c[0])

    def trailer_item_tags(self, c):
        return ("tags", c[0])

    def trailer_item_hints(self, c):
        return ("hints", c[0])

    def trailer(self, c):
        out = {"note": "", "hints": ""}
        seen = set()
        for key, value in c:
            if key in seen:
                raise ValueError(
                    f"repeated {key}{{...}} clause: at most one of each of "
                    f"note, tags, hints is allowed per statement.")
            seen.add(key)
            out[key] = value
        return out

    # ----- exposures -------------------------------------------------
    @staticmethod
    def _check_fyi_premium(numbers, fyi, head):
        """Guard the informational-premium suffix ([FYI-Premium-Exposure-Head]).

        An FYI premium is one booked number for the aggregate and must never
        touch the law or the per-component reporting, so both the sizing amount
        and the premium are required to be scalar. A vector premium against a
        scalar head would broadcast into extra components (changing the
        distribution); a scalar premium against a vector head would repeat per
        component and misreport the total. Both are refused here rather than
        surprising downstream.
        """
        if not fyi:
            return
        if not np.isscalar(numbers):
            raise ValueError(
                f'DecL: an FYI premium requires a scalar {head} amount, '
                f'not a vector (the premium would repeat across the '
                f'exposure components and misreport the total)')
        if not np.isscalar(fyi["exp_premium"]):
            raise ValueError(
                'DecL: an FYI premium must be a scalar, not a vector '
                '(a vector premium would broadcast into extra components '
                'and change the distribution)')

    def fyi_premium_some(self, c):
        # ``<amount> premium [as <label>]`` after a claims / loss head: the
        # booked (informational) premium. Scalar-only, enforced by
        # ``_check_fyi_premium`` in the head that receives it.
        numbers, _premium, as_label = c
        return {"exp_premium": numbers,
                "_premium_label": as_label.get("label")}

    def fyi_premium_none(self, c):
        return {}

    def exposures_claims(self, c):
        numbers, _claims, as_label, fyi = c
        self._check_fyi_premium(numbers, fyi, 'claims')
        return {"exp_en": numbers,
                "_exposure_label": as_label.get("label"),
                **fyi}

    def exposures_loss(self, c):
        numbers, _loss, as_label, fyi = c
        self._check_fyi_premium(numbers, fyi, 'loss')
        return {"exp_el": numbers,
                "_exposure_label": as_label.get("label"),
                **fyi}

    def exposures_premium_lr(self, c):
        prem, _premium, as_label, _at, lr, _lr = c
        return {
            "exp_premium": prem,
            "exp_lr": lr,
            "exp_el": np.array(prem) * np.array(lr),
            "_exposure_label": as_label.get("label"),
        }

    def exposures_exposure_rate(self, c):
        exp_, _exposure, as_label, _at, rate, _rate = c
        return {
            "exp_premium": exp_,
            "exp_lr": rate,
            "exp_el": np.array(exp_) * np.array(rate),
            "_exposure_label": as_label.get("label"),
        }

    @staticmethod
    def _check_years_scalar(value, what):
        """Renewal exposure terms are scalars -- there is one clock."""
        if not np.isscalar(value):
            raise ValueError(
                f'DecL: {what} in a years (renewal) exposure must be a '
                f'scalar, not a vector')
        return float(value)

    def exposures_years(self, c):
        numbers, _years, as_label = c
        return {"exp_years": self._check_years_scalar(numbers, 'years'),
                "_exposure_label": as_label.get("label")}

    def exposures_years_rate(self, c):
        # ``T years at r rate``: informational premium T*r (feeds PnL gross
        # premium / lr); the claim count comes solely from the wait law.
        # ``exp_rate`` is stored for byte-exact writer round-trip.
        years, _years, as_label, _at, rate, _rate = c
        years = self._check_years_scalar(years, 'years')
        rate = self._check_years_scalar(rate, 'the rate')
        return {"exp_years": years,
                "exp_rate": rate,
                "exp_premium": years * rate,
                "_exposure_label": as_label.get("label")}

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
        if bid.get("freq_name") == "renewal":
            # the count of a renewal aggregate comes solely from the wait
            # law over the fixed horizon -- exposure cannot scale it (and
            # exp_en is the -1 empirical sentinel, not a count)
            logger.warning(
                "'@' inhomogeneous scaling is a frequency no-op for a "
                "renewal (wait-clause) aggregate %s; spec left unscaled",
                bid.get("name", ""))
            return bid
        bid["exp_en"] = _check_vectorizable(bid.get("exp_en", 0)) * expr
        bid["exp_el"] = _check_vectorizable(bid.get("exp_el", 0)) * expr
        bid["exp_premium"] = _check_vectorizable(bid.get("exp_premium", 0)) * expr
        return bid

    @staticmethod
    def _refuse_sev_ref_algebra(bid, operation):
        """Refuse severity algebra on an aggregate whose severity is a reference.

        The homogeneous and shift forms rewrite ``sev_mean`` / ``sev_scale`` /
        ``sev_loc``, none of which a deferred ``agg.NAME`` reference has: the
        severity is another object's output law, and what scaling or shifting
        that law would mean is undecided. Silently scaling only the exposure
        keys would be worse than refusing. ``@`` (inhomogeneous) is frequency
        only and stays legal.
        """
        if "sev_ref" in bid:
            raise ValueError(
                f"DecL: {bid.get('name', 'the aggregate')} takes its severity "
                f"from the reference '{bid['sev_ref']}', which cannot be "
                f"{operation}: a reference is another object's output law, not "
                "a parameterized family. Scale or shift the referenced "
                "declaration itself, or use '@' to scale the exposure.")

    def builtin_agg_homog(self, c):
        expr, _times, bagg = c
        bid = bagg
        self._refuse_sev_ref_algebra(bid, 'scaled')
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
        self._refuse_sev_ref_algebra(bid, 'shifted')
        bid["name"] += "_shifted"
        if "sev_loc" in bid:
            bid["sev_loc"] += expr
        else:
            bid["sev_loc"] = expr
        return bid

    def builtin_agg_minus(self, c):
        bagg, _minus, expr = c
        bid = bagg
        self._refuse_sev_ref_algebra(bid, 'shifted')
        bid["name"] += "_shifted"
        if "sev_loc" in bid:
            bid["sev_loc"] -= expr
        else:
            bid["sev_loc"] = -expr
        return bid

    def builtin_agg_lookup(self, c):
        return self.safe_lookup(c[0])

    # ----- name + optional display label -----------------------------
    def name(self, c):
        return c[0]

    def as_label_some(self, c):
        # c = [AS token, label value]; surfaced as a spec key that objects merge
        # (``**as_label``) and the premium head / expense group read out.
        return {"label": c[1]}

    def as_label_none(self, c):
        return {}

    def label_id(self, c):
        # a bareword label (``as lae``) -- skips the quote tax.
        return str(c[0])

    def label_string(self, c):
        # a quoted label (``as "Loss Adjustment Expense"``) -- strip the delimiters.
        return str(c[0])[1:-1]

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
    # ?expr / ?term / ?factor / ?sum / ?product are inlined in the grammar,
    # so the transformer only sees the aliased nodes below.
    #
    # `+`, `-` and `*` are legal only inside parentheses (the paren island,
    # ?sum / ?product in decl.lark); a bare expression sees only `/`, `**`,
    # `^`, `exp` and parentheses. Everything evaluates here at parse time,
    # so the spec carries a plain float and the canonical decompiled text
    # shows the evaluated literal.
    #
    # A _PercentNumber survives literal use only, by existing design: any
    # arithmetic returns a plain float, so a computed value in the `po`
    # placement position reads as an absolute amount, not a percentage.

    def atom_add(self, c):
        a, _, b = c
        return a + b

    def atom_subtract(self, c):
        a, _, b = c
        return a - b

    def atom_multiply(self, c):
        a, _, b = c
        return a * b

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
    ``'agg'``, ``'sev'``, ``'port'``, ``'bvagg'``, ``'pnl'``, ``'xpnl'``,
    ``'distortion'``, or ``'expr'``."""

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
            ``kind`` is one of ``'agg'``, ``'sev'``, ``'port'``, ``'bvagg'``,
            ``'pnl'``, ``'xpnl'``, ``'distortion'``, ``'expr'``; ``name`` is
            the object identifier; ``spec`` is the dictionary specification
            used downstream to construct the object. For ``'expr'`` the
            statement is a bare expression and ``spec`` is its numeric value.

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
        ref_include.rst`` wrapped in a Sphinx ``code-block:: text`` directive
        so it can be ``include``-d by the language reference. The language is
        ``text`` because Pygments ships no ``lark`` lexer, and an unknown name
        is a build warning.
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
        out.write_text(f".. code-block:: text\n\n{body}\n", encoding="utf-8")

    target = Path(save_to_fn) if save_to_fn else Path.home() / "aggregate/parser/grammar.lark"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    return text


if __name__ == "__main__":
    grammar(add_to_doc=True)
