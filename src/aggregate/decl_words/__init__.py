"""
The DecL word list: every word a program needs to color DecL, in one place.

:func:`decl_words` returns the words by group. The Pygments lexer
(:class:`aggregate.decl_pygments.AggLexer`) builds its keyword rules from it,
and programs outside Python read the same list as JSON from
``python -m aggregate.decl_words``. Adding a keyword to ``decl.lark`` then
colors it everywhere, with no copy to update by hand.

Submodule access only: ``from aggregate.decl_words import decl_words``. The
function is deliberately not re-exported from :mod:`aggregate`, where it would
shadow this module's name.

Notes
-----
**Where each group comes from.** Four groups are read from ``decl.lark`` when
the list is first built, so they cannot drift from the grammar:

- ``reserved``: the ``ID`` terminal's exclusion alternation, the flat list of
  every word the grammar refuses as an identifier.
- ``frequencies``: the ``FREQ.2`` alternation.
- ``clauses``: the ``word\\{`` terminals (``note``, ``hints``, ``tags``). They
  are not reserved, because their terminals include the opening brace and so
  the bare word is a legal identifier.
- ``constants``: the word spelled in ``NUMBER.2`` (``inf``).

One group comes from a registry: ``distortions`` is
``Distortion.available_distortions(pricing=False)``, every documented kind.

Three groups are kept here by hand, because the grammar does not hold the words:

- ``declarations``: the discrete declaration keywords. Each is a reserved word;
  the group separates them from the other keywords because a colorizer gives
  them their own color.
- ``severities``: severity distribution names, mapped to their shape parameter
  count. The grammar reads a severity name as a plain ``ID`` and the library
  resolves it with ``getattr(scipy.stats, name)``, so no file enumerates them.
  ``cantor`` is the library's own (one shape), and ``dhistogram`` and
  ``chistogram`` are the empirical severities, which take no shape parameter
  and map to ``None``. The tests check every name resolves and every count
  matches scipy.
- ``mixing``: the mixing distributions accepted after ``mixed``, checked by the
  tests against the frequency registry.

**One word, several groups.** ``beta`` is a severity, a mixing distribution and a
distortion kind. A colorizer that reads words without their grammatical
context cannot tell ``sev beta`` from ``distortion beta``, so the rule for every
consumer is: a word in any other group is a keyword, and only a word found
solely in ``distortions`` is colored as a distortion name.
"""

from __future__ import annotations

import functools
import re

from ..parser import GRAMMAR_FILE

__all__ = ['decl_words']


#: The discrete declaration keywords, each a reserved word in the grammar.
_DECLARATIONS = ('dbvsev', 'dfreq', 'dsev', 'dwait', 'elt', 'nttagg', 'ylt')

#: Severity distribution names mapped to their shape parameter count. Every
#: name but the three the library special cases (``cantor``, ``dhistogram``,
#: ``chistogram``) is a ``scipy.stats`` continuous distribution, and its count is
#: that distribution's ``numargs``.
_SEVERITIES = {
    # No shape parameter.
    'anglit': 0, 'arcsine': 0, 'cauchy': 0, 'cosine': 0, 'expon': 0,
    'gibrat': 0, 'gumbel_l': 0, 'gumbel_r': 0, 'halfcauchy': 0,
    'halflogistic': 0, 'halfnorm': 0, 'hypsecant': 0, 'kstwobign': 0,
    'laplace': 0, 'levy': 0, 'levy_l': 0, 'logistic': 0, 'maxwell': 0,
    'moyal': 0, 'norm': 0, 'rayleigh': 0, 'semicircular': 0, 'uniform': 0,
    'wald': 0,
    # One shape parameter.
    'alpha': 1, 'argus': 1, 'bradford': 1, 'cantor': 1, 'chi': 1, 'chi2': 1,
    'dgamma': 1, 'dweibull': 1, 'erlang': 1, 'exponnorm': 1, 'exponpow': 1,
    'fatiguelife': 1, 'fisk': 1, 'foldcauchy': 1, 'foldnorm': 1, 'gamma': 1,
    'genextreme': 1, 'genhalflogistic': 1, 'genlogistic': 1, 'gennorm': 1,
    'genpareto': 1, 'gompertz': 1, 'halfgennorm': 1, 'invgamma': 1,
    'invgauss': 1, 'invweibull': 1, 'kappa3': 1, 'ksone': 1, 'kstwo': 1,
    'laplace_asymmetric': 1, 'loggamma': 1, 'loglaplace': 1, 'lognorm': 1,
    'lomax': 1, 'nakagami': 1, 'pareto': 1, 'pearson3': 1, 'powerlaw': 1,
    'powernorm': 1, 'rdist': 1, 'recipinvgauss': 1, 'rice': 1,
    'skewcauchy': 1, 'skewnorm': 1, 't': 1, 'triang': 1, 'truncexpon': 1,
    'tukeylambda': 1, 'vonmises': 1, 'vonmises_line': 1, 'weibull_max': 1,
    'weibull_min': 1, 'wrapcauchy': 1,
    # Two shape parameters.
    'beta': 2, 'betaprime': 2, 'burr': 2, 'burr12': 2, 'crystalball': 2,
    'exponweib': 2, 'f': 2, 'gengamma': 2, 'geninvgauss': 2, 'johnsonsb': 2,
    'johnsonsu': 2, 'kappa4': 2, 'levy_stable': 2, 'loguniform': 2,
    'mielke': 2, 'nct': 2, 'ncx2': 2, 'norminvgauss': 2, 'powerlognorm': 2,
    'reciprocal': 2, 'studentized_range': 2, 'trapezoid': 2, 'truncnorm': 2,
    # Empirical: the outcomes and probabilities follow, there is no shape.
    'chistogram': None, 'dhistogram': None,
}

#: The mixing distributions accepted after ``mixed``.
_MIXING = ('beta', 'delaporte', 'gamma', 'ig', 'sichel', 'sichel.gamma',
           'sichel.ig', 'sig')


def _grammar_text() -> str:
    """The text of the shipped ``decl.lark``."""
    return GRAMMAR_FILE.read_text(encoding='utf-8')


def _grammar_line(text: str, prefix: str) -> str:
    """The one line of ``text`` that starts with the terminal ``prefix``."""
    return next(line for line in text.splitlines()
                if line.lstrip().startswith(prefix))


def _grammar_reserved_words(text: str | None = None) -> tuple[str, ...]:
    """Every reserved word, read from the ``ID`` terminal's exclusion list.

    Parameters
    ----------
    text : str, optional
        The text of ``decl.lark``; read from the package when omitted.

    Returns
    -------
    tuple of str
        Sorted reserved words.

    Notes
    -----
    The ``ID`` terminal rejects bare keywords through a single negative
    lookahead alternation, the authoritative flat list of every reserved word::

        ID: /(?!agg\\.|...)(?!(?:agg|aggregate|...|swing)(?![namechar])).../
    """
    text = _grammar_text() if text is None else text
    m = re.search(r"\(\?!\(\?:([a-zA-Z0-9_|]+)\)\(\?!", _grammar_line(text, 'ID:'))
    if m is None:
        raise ValueError('decl.lark: no keyword exclusion list in the ID terminal')
    return tuple(sorted(set(m.group(1).split('|'))))


def _grammar_frequencies(text: str | None = None) -> tuple[str, ...]:
    """The frequency distribution names, read from the ``FREQ.2`` alternation.

    Parameters
    ----------
    text : str, optional
        The text of ``decl.lark``; read from the package when omitted.

    Returns
    -------
    tuple of str
        Sorted frequency names.
    """
    text = _grammar_text() if text is None else text
    m = re.search(r"\(\?:([a-zA-Z0-9_|]+)\)", _grammar_line(text, 'FREQ.2:'))
    if m is None:
        raise ValueError('decl.lark: no alternation in the FREQ terminal')
    return tuple(sorted(set(m.group(1).split('|'))))


def _grammar_brace_clauses(text: str | None = None) -> tuple[str, ...]:
    """The trailer clause words, read from the ``word\\{`` terminals.

    Parameters
    ----------
    text : str, optional
        The text of ``decl.lark``; read from the package when omitted.

    Returns
    -------
    tuple of str
        Sorted clause words.

    Notes
    -----
    Derived rather than written down, so a clause added to ``decl.lark`` reaches
    every colorizer. A hand list is how ``tags`` and ``doc`` once went uncolored
    for several releases.
    """
    text = _grammar_text() if text is None else text
    found = re.findall(r"^([A-Z]+)\.\d+:\s*/([a-z]+)\\\{", text, re.MULTILINE)
    return tuple(sorted({word for _terminal, word in found}))


def _grammar_constants(text: str | None = None) -> tuple[str, ...]:
    """The words spelled inside the ``NUMBER.2`` terminal (today, ``inf``).

    Parameters
    ----------
    text : str, optional
        The text of ``decl.lark``; read from the package when omitted.

    Returns
    -------
    tuple of str
        Sorted constant words.

    Notes
    -----
    A word here is a run of two or more lowercase letters not preceded by a
    backslash or another letter, which skips the regex escapes (``\\d``) and the
    exponent marker (``eE``) in the numeric alternatives.
    """
    text = _grammar_text() if text is None else text
    pattern = _grammar_line(text, 'NUMBER.2:')
    return tuple(sorted(set(re.findall(r'(?<![\\a-zA-Z])[a-z]{2,}', pattern))))


@functools.cache
def _build() -> dict:
    """Build the groups once. See :func:`decl_words`."""
    from ..spectral import Distortion

    text = _grammar_text()
    return {
        'reserved': _grammar_reserved_words(text),
        'frequencies': _grammar_frequencies(text),
        'declarations': _DECLARATIONS,
        'clauses': _grammar_brace_clauses(text),
        'constants': _grammar_constants(text),
        'severities': dict(sorted(_SEVERITIES.items())),
        'mixing': _MIXING,
        'distortions': tuple(sorted(Distortion.available_distortions(pricing=False))),
    }


def decl_words() -> dict:
    """Every DecL word a colorizer needs, by group.

    Returns
    -------
    dict
        Group name to words. Every group is a sorted tuple of str except
        ``severities``, a dict of name to shape parameter count, sorted by name,
        with ``None`` for the empirical severities. The groups are
        ``reserved``, ``frequencies``, ``declarations``, ``clauses``,
        ``constants``, ``severities``, ``mixing`` and ``distortions``.

    Notes
    -----
    Built on the first call and cached; each call returns a fresh outer dict and
    a fresh ``severities`` dict, so a caller may change what it receives. The
    module docstring says which groups come from ``decl.lark``, which from the
    distortion registry and which are kept by hand, and gives the rule for a
    word, such as ``beta``, that sits in more than one group.

    Examples
    --------
    >>> from aggregate.decl_words import decl_words
    >>> decl_words()['clauses']
    ('hints', 'note', 'tags')
    """
    return {group: dict(words) if isinstance(words, dict) else words
            for group, words in _build().items()}
