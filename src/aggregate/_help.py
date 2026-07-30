"""``HelpMixin`` -- the shared ``.help(regex)`` discovery surface.

``.help('regex')`` is the front door to every first-class object in the library:
it lists the public methods and properties whose names match ``regex``, so a user
who knows *what* they want but not *what it is called* can find it from the
object itself. That makes it the one method every class should carry. ``regex``
defaults to ``'.*'``, so ``.help()`` with no argument lists everything, which is
where a user with no name in mind starts.

It is a single delegation to :func:`aggregate.utilities.agg_help`, and before
this mixin that one line -- plus an identical dozen-line docstring -- was
copy-pasted into nine classes across eight modules. Two classes that should have
had it did not: :class:`~aggregate._frequency.Frequency` and
:class:`~aggregate._grid_distribution.GridDistribution`. Collapsing the copies
here closed both holes for free, which is the argument for the mixin: a surface
every class must share should be defined once.

Notes
-----
Like :class:`~aggregate._labeled.LabeledMixin`, this defines **no** ``__init__``
-- it stays transparent to each host's ``super()`` chain, which matters because
the hosts have wildly different bases (``Severity`` extends
``scipy.stats.rv_continuous``, ``Portfolio`` extends ``object``). It holds no
state at all: ``help`` reads ``self`` by introspection and returns nothing.

:func:`~aggregate.utilities.agg_help` is imported **inside** the method, not at
module scope. ``utilities`` imports ``_grid_distribution`` at import time, and
``_grid_distribution`` is one of the hosts -- a module-level import here would
close the cycle ``utilities -> _grid_distribution -> _help -> utilities`` and
fail, because ``agg_help`` is not yet bound when ``utilities`` is only part-way
through its own module body.
"""


class HelpMixin:
    """Mixin supplying the universal ``.help(regex, ...)`` discovery method."""

    def help(self, regex='.*', lod='terse', values='none', private=False, fmt='auto'):
        """
        Lookup help on methods and properties matching ``regex``.

        Thin wrapper over :func:`aggregate.utilities.agg_help` -- the free
        function is prefixed to avoid shadowing Python's builtin ``help`` at
        module / package scope. Called with no arguments, ``regex`` defaults to
        ``'.*'``, which matches every name: a bare ``.help()`` lists the whole
        public surface. Four axes: ``lod``
        (``'terse'|'short'|'all'``) controls how much docstring is shown;
        ``values`` (``'none'|'short'|'all'``) how much of each value or
        no-argument call result (a ``DataFrame`` / ``Series`` is headed to 5
        rows under ``'short'``); ``private`` (``False``) whether ``_``-prefixed
        names are included; ``fmt`` (``'auto'|'text'|'ansi'|'html'``) the
        render target (``auto`` = ANSI in Jupyter, plain text in a terminal).
        The default ``lod='terse', values='none', private=False`` is a bare
        public-name listing.

        Parameters
        ----------
        regex : str, default '.*'
            Regular expression matched against public member names. The default
            matches everything.
        lod : {'terse', 'short', 'all'}, default 'terse'
            Level of documentation detail per match.
        values : {'none', 'short', 'all'}, default 'none'
            How much of each member's value or call result to show.
        private : bool, default False
            Include ``_``-prefixed names.
        fmt : {'auto', 'text', 'ansi', 'html'}, default 'auto'
            Render target.

        Returns
        -------
        None
            Output is printed / displayed, not returned.
        """
        from .utilities import agg_help     # deferred: see the module docstring
        agg_help(self, regex, lod=lod, values=values, private=private, fmt=fmt)
