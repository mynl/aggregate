"""Shared setup for the aggregate cookbook — see ``plan.md`` [Cookbook-Mechanics].

Every cookbook page begins with ``from _setup import *``. The import is
idempotent, so the master ``cookbook.qmd`` re-running it once per Quarto
``{{< include >}}`` costs nothing; and each page therefore runs **standalone in
JupyterLab**.

This module is the **single source of truth for example calibration**
([Cookbook-Open-Questions] #3): the house-book parameters and the reusable DecL
snippets live here, and pages compose them. A page's beat 1 still pretty-prints
the fully-resolved program (via :func:`pp`), so it stays self-explanatory even
though the numbers are centralized here. **Calibration is PROVISIONAL — the
author tunes it.**

Tables render through :func:`qd`, the generic quick-display verb, wired here to
``greater_tables.GT`` for crisp HTML ([Cookbook-Open-Questions] #2). It degrades
to a plain ``display`` if ``greater_tables`` is not installed.
"""
from IPython.display import Markdown, display

from textwrap import fill

from aggregate import build, format_program
from aggregate import __version__ as version

try:  # public re-export preferred; fall back to the constants module
    from aggregate import IgnoredDecLClauseWarning
except ImportError:  # pragma: no cover
    from aggregate.constants import IgnoredDecLClauseWarning

try:  # the author's table library — much clearer HTML than the pandas default
    from greater_tables import GT
    _HAVE_GT = True
except Exception:  # pragma: no cover
    _HAVE_GT = False

import pandas as pd
import numpy as np
import warnings

__all__ = [
    'build', 'format_program', 'version', 'IgnoredDecLClauseWarning',
    'warnings', 'display', 'pd',
    'qd', 'cbqd', 'pp', 'show', 'recipe',
    # calibrated DecL library (the source of truth)
    'HOUSE_PREM', 'HOUSE_SEV', 'HOUSE_FREQ', 'house',
    'OCC_LAYER', 'AGG_LAYER', 'SWING', 'REINST', 'pnl',
]

# --------------------------------------------------------------------------
# display helpers
# --------------------------------------------------------------------------

def qd(x, **kwargs):
    """Generic quick-display, wired to ``greater_tables.GT`` for HTML.

    Pass a DataFrame or Series; ``None`` is a no-op (handy with ``getattr``
    fallbacks). Falls back to a plain ``display`` when ``greater_tables`` is
    absent, so the cookbook still renders anywhere.
    """
    if x is None:
        return
    if _HAVE_GT and isinstance(x, (pd.DataFrame, pd.Series)):
        display(GT(x.to_frame() if isinstance(x, pd.Series) else x, **kwargs))
    else:
        display(x)


def recipe(name, *, uw=None, run=True, show_check=True):
    """Render one library recipe: Problem, Solution (with output), Discussion, Check.

    **The one cookbook verb.** A recipe page is a heading plus a call to this;
    everything else comes from the entry's ``doc{{{...}}}`` in the shipped
    library, so the page and the pytest harness read the same source and cannot
    drift apart.

    Output is emitted through ``IPython.display`` rather than Quarto's
    ``#| output: asis``, deliberately: asis text and rich display output do not
    interleave reliably within one cell, and a recipe needs its tables and plots
    to appear *between* its prose sections, in order.

    Parameters
    ----------
    name : str
        Library entry name, the same string you would pass to ``build``.
    uw : Underwriter, optional
        Knowledge base to read from; the default ``build`` underwriter by
        default.
    run : bool, default True
        Execute the Solution and Check. ``False`` renders the prose and the
        code without running anything — useful for an expensive recipe on a
        page you want to render fast.
    show_check : bool, default True
        Render the Check block. It is the point of the cookbook, so this
        defaults on; pass ``False`` for a page where the invariant is discussed
        in prose instead.

    Returns
    -------
    dict or None
        The namespace the recipe's code left behind (``None`` when
        ``run=False``), so a page can carry on from where the recipe stopped.
    """
    uw = uw or build
    r = uw.recipe(name)

    display(Markdown(f'**Problem.** {r.problem}' if r.problem
                     else f'*(no Problem section for `{name}`)*'))

    # Beat 1: the declaration itself, pretty-printed, trailer suppressed --
    # the note/tags/doc are what produced this page, not part of the example.
    display(Markdown('**Solution.**\n\n' + (r.solution or '')))
    if r.program:
        display(Markdown(f'```\n{format_program(r.program, fmt="text")}\n```'))

    ns = None
    if run and r.is_runnable:
        ns = r.namespace()
        exec(compile(r.solution_code, f'<recipe {name}:solution>', 'exec'), ns)

    if r.discussion:
        display(Markdown(f'**Discussion.** {r.discussion}'))

    if show_check and r.check:
        display(Markdown(
            '::: {.callout-note collapse="true" title="The check"}\n\n'
            f'{r.check}\n\n:::'))
        if run and r.check_code:
            exec(compile(r.check_code, f'<recipe {name}:check>', 'exec'),
                 ns if ns is not None else r.namespace())
    return ns


def cbqd(ob):
    """Cookbook qd."""
    print(ob.format_program(fmt='text', trailer=False))
    print()
    print(fill(ob._text_info_blob(), 65))
    print()
    qd(ob.summary_df.fillna(''))
    print()
    if ob.note != '':
        print(f'Note: {ob.note}')


def pp(decl):
    """Beat 1 -- pretty-print a DecL program in color, aiming for self-explanatory."""
    print(format_program(decl, fmt='text'))


def show(ob, *, summary=True, plot=True):
    """Beats 2-3 -- the common first-class-citizen surface.

    Prints ``valid`` / ``validation_explanation`` (beat 2), then displays
    ``summary_df`` via :func:`qd` and a density plot (beat 3). Deliberately lean:
    a page adds its own extra exhibits (``stats_df``, ``reins_summary_df``, an
    ``*_explanation``) as the recipe warrants.
    """
    name = getattr(ob, 'name', type(ob).__name__)
    valid = getattr(ob, 'valid', None)
    if valid is not None:
        vexp = getattr(ob, 'validation_explanation', '')
        print(f'{name}: valid={valid}  ({vexp})')
    if summary:
        qd(getattr(ob, 'summary_df', None))
    if plot and hasattr(ob, 'plot'):
        try:
            ob.plot()
        except Exception as exc:  # keep a page rendering even if a plot chokes
            print(f'(plot skipped: {exc})')
    return ob


# --------------------------------------------------------------------------
# the calibrated DecL library -- ONE house book + reusable fragments, reused
# across pages. PROVISIONAL calibration; the author tunes these numbers.
# --------------------------------------------------------------------------

# gross exposure: plan premium at a target loss ratio, a basic limit, a
# thick-tailed single-parameter Pareto severity (unlimited -> slognorm-ish).
HOUSE_PREM = '10000 prem at 75% lr'
HOUSE_SEV = '5000 xs 0 sev 100 * pareto 0.75 - 100'
HOUSE_FREQ = 'mixed gamma 0.25'

# reusable reinsurance / variable-rating clauses, calibrated to the house book.
# DecL clause order is fixed: occurrence reins goes BEFORE the frequency clause,
# aggregate reins AFTER it -- so OCC_LAYER feeds house(occ=...) and AGG_LAYER
# feeds house(agg=...). Neither carries a frequency clause (house supplies it).
OCC_LAYER = 'occurrence net of 4000 xs 1000 rol 8%'
AGG_LAYER = 'aggregate net of 5000 xs 10000 deposit 1250'
SWING = 'swing basic 500 lcm 1.5 min 500 max 3000'  # goes inside an aggregate layer
REINST = 'reinstatements [1 1 1]'                    # follows an occurrence layer


def house(name='House', *, occ='', freq=HOUSE_FREQ, agg='', tail=''):
    """A calibrated gross book as a DecL string, ready to extend with reinsurance.

    Assembles the clauses in DecL's required order:
    ``agg NAME <premium> <severity> [occ reins] <freq> [agg reins] [tail]``.

    Parameters
    ----------
    name : str
        Aggregate name.
    occ : str
        Occurrence reinsurance clause (placed before frequency), e.g. ``OCC_LAYER``.
    freq : str
        Frequency clause (defaults to the house mixed-gamma).
    agg : str
        Aggregate reinsurance clause (placed after frequency), e.g. ``AGG_LAYER``.
    tail : str
        Trailing clauses (expenses, ...) appended last.

    Returns
    -------
    str
        A complete ``agg ...`` DecL program.
    """
    parts = ['agg', name, HOUSE_PREM, HOUSE_SEV, occ, freq, agg, tail]
    return ' '.join(p for p in parts if p).strip()


def pnl(engine, name='Book', *, walk=False):
    """Wrap a complete engine (an ``agg ...`` string) in a ``pnl`` / ``xpnl``.

    ``walk=True`` gives the step-by-step ``xpnl``; otherwise the consolidated
    ``pnl``. The engine inherits its premium from the wrapped book.
    """
    kind = 'xpnl' if walk else 'pnl'
    return f'{kind} {name} inherit premium less {engine}'
