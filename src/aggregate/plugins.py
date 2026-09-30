"""``aggregate.plugins``: the extension surface a third-party package codes against.

.. warning::

   **Provisional, in the sense of PEP 411**, and in fact weaker than that. This
   module exists to let out-of-tree packages register into
   :mod:`aggregate.charts` and :mod:`aggregate.exhibits`, both of which are
   themselves provisional, so anything registering through here inherits their
   freedom to change in a minor release with no deprecation period. See
   :doc:`/3_reference/3_x_API_Stability`.

"Lock the 1.0 API down" and "allow extras" are the same design act, not a
contradiction: the **consumption** surface users code against is frozen, and a
separate, explicitly unstable **extension** surface is published for plugin
authors. This module is that second surface.

A plugin is an ordinary distribution named ``aggregate-<thing>`` with an import
package ``aggregate_<thing>``. It declares one entry point::

    [project.entry-points."aggregate.plugins"]
    relativity = "aggregate_relativity.register:register"

and that entry point resolves to a zero-argument ``register()`` callable which
performs its :func:`~aggregate.charts.register_chart` and
:func:`~aggregate.exhibits.register_simple_exhibit` calls and returns a list of
:class:`PluginLeaf`, the app-facing strings for what it just registered.

Plugins live in their **own distributions**. The library carries no ``contrib``
bag: ``extensions/`` was exactly that and was deleted at ``1.0.0a12``, because
experimental code inside the library becomes the library's dependency surface,
its test burden and its stability question, and a locked API shipped beside a
``contrib`` bag is theater.

Four rules worth knowing before writing one.

**Nothing loads on import.** ``import aggregate`` stays deterministic; the
**host** calls :func:`load`. A notebook's results should be a function of the
notebook's own text and not of what happens to be installed, and the pytest
suite must not change behavior because a sibling package was synced. A server
is a deployment and a deployment may declare what it trusts, so the API server
calls :func:`load` in ``create_app()`` behind a settings flag.

**Provenance is captured, never parsed from names.** :func:`load` snapshots the
registry key sets before and after each plugin's ``register()`` and records the
difference, so no naming convention is load bearing and a plugin may name its
chart ``relativity`` rather than ``gini.relativity``.

**A failing plugin is recorded, not fatal.** An import or a ``register()`` that
raises is caught, its traceback stored on the plugin's :class:`LoadedPlugin`,
and loading continues. One broken experiment must never take down
:func:`~aggregate.build` or the server.

**Name reuse across plugins is refused, not merged.**
:func:`~aggregate.charts.register_chart` already raises on a duplicate name, so
a chart collision arrives here as a recorded failure.
:func:`~aggregate.exhibits.register_simple_exhibit` deliberately does the
opposite: calling it twice for one name *extends* the existing exhibit to more
classes, which is correct inside the library's own manifest (that is how one
exhibit carries a different caption per class) and wrong across a trust
boundary, where it would silently merge two unrelated plugins into one exhibit.
This module therefore polices the case itself and leaves
``register_simple_exhibit`` untouched.

Notes
-----
There is no removal API on either registry and this module does not add one, so
a rejected plugin is **not** unwound: whatever it managed to register before the
rejection stays registered. A collision is a packaging mistake to fix, and the
process warrants a restart once it is fixed.

Discovery reads entry points once, at the :func:`load` call. A newly installed
plugin needs a fresh process; ``uvicorn --reload`` covers the experiment loop.

Two environment variables are read here rather than through
:mod:`aggregate.config`, since neither is a setting that changes a numerical
answer and :func:`load` must work before any settings object exists. Both are
named in ``config._ENV_IGNORED`` so the ``AGGREGATE_*`` allow-list does not warn
about them.
"""

import os
import traceback
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, entry_points, version

__all__ = [
    'ENTRY_POINT_GROUP', 'ENV_NO_PLUGINS', 'ENV_PLUGINS', 'LEAF_KINDS',
    'LoadedPlugin', 'PluginLeaf', 'load', 'loaded_plugins', 'reset_plugins',
]

#: The entry point group a plugin distribution declares.
ENTRY_POINT_GROUP = 'aggregate.plugins'

#: Comma-separated importable module names, the zero-ceremony experiment route.
#: Each module supplies the ``register`` callable, so the value that pairs with
#: the entry point above is ``aggregate_relativity.register``.
ENV_PLUGINS = 'AGGREGATE_PLUGINS'

#: Set truthy to suppress discovery entirely, whatever else is installed.
ENV_NO_PLUGINS = 'AGGREGATE_NO_PLUGINS'

#: What a leaf can be. A plugin contributes documents, never app behavior: no
#: forms, no input controls, nothing that needs code in the client. That
#: boundary is the entire reason the mechanism is cheap.
LEAF_KINDS = ('chart', 'exhibit')

_FALSEY = frozenset({'', '0', 'false', 'no', 'off'})


@dataclass(frozen=True)
class PluginLeaf:
    """One document a plugin contributes, and the strings a client shows for it.

    The plugin authors every string. They deliberately do **not** live on
    :class:`~aggregate.charts.ChartEntry` or on an
    :class:`~aggregate.exhibits.Exhibit`: those are IR concerns, a navigation
    hint is a client concern, and mixing the two puts UI strings in the
    library's chart registry forever.

    Parameters
    ----------
    name : str
        The registry key, the same string :func:`~aggregate.charts.register_chart`
        or :func:`~aggregate.exhibits.register_simple_exhibit` was given.
    kind : str
        ``'chart'`` or ``'exhibit'``, one of :data:`LEAF_KINDS`.
    label : str
        The leaf's display name.
    hint : str, optional
        One line saying what the document shows, for a tooltip.
    why : str, optional
        Why the leaf is dark when it cannot serve an object, shown verbatim.
        Nothing is hidden, so a leaf that cannot answer says why not.
    """

    name: str
    kind: str
    label: str
    hint: str = ''
    why: str = ''

    def __post_init__(self):
        if self.kind not in LEAF_KINDS:
            raise ValueError(
                f'PluginLeaf kind {self.kind!r} is not one of {LEAF_KINDS}')


@dataclass(frozen=True)
class LoadedPlugin:
    """The record of one discovered plugin, whether or not it loaded.

    Parameters
    ----------
    name : str
        The entry point name, or for the environment route the module path as
        it was written, since there is no distribution metadata to name it.
    version : str or None
        The distribution version where one is discoverable.
    source : str
        ``'entry_point'`` or ``'env'``.
    leaves : tuple of PluginLeaf
        What ``register()`` declared, in the order it declared it. Empty on a
        failure.
    charts : tuple of str
        The chart registry keys this plugin actually added, from the
        before-and-after snapshot rather than from anything it claimed.
    exhibits : tuple of str
        The exhibit registry keys it actually added, likewise.
    error : str or None
        ``None`` on success, otherwise the recorded traceback or the rejection
        message. A client showing a one-line summary should take the last line.
    """

    name: str
    version: str | None
    source: str
    leaves: tuple = ()
    charts: tuple = ()
    exhibits: tuple = ()
    error: str | None = None

    @property
    def ok(self):
        """True when the plugin loaded and registered without complaint."""
        return self.error is None


@dataclass
class _Discovered:
    """A candidate found by discovery, before its ``register()`` has run."""

    name: str
    source: str
    version: str | None = None
    entry_point: object = None
    module_path: str | None = None
    leaves: list = field(default_factory=list)


_LOADED = []
_DONE = False


def _truthy(value):
    """True unless the environment string reads as off."""
    return value is not None and value.strip().lower() not in _FALSEY


def _distribution_version(dist_name):
    """The installed version of ``dist_name``, or None when it is not a package."""
    try:
        return version(dist_name)
    except (PackageNotFoundError, ValueError):
        return None


def _discover(env):
    """Return the candidate plugins, ordered by name, deduplicated.

    Entry points come first for a given name, since the packaged route carries
    dependency management and versioning and the environment route is the
    experiment. Ordering is by plugin name so leaf order is stable across
    installs and does not depend on the order ``importlib.metadata`` happens to
    return distributions in.
    """
    found = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in found:
            continue
        dist = getattr(ep, 'dist', None)
        found[ep.name] = _Discovered(
            name=ep.name, source='entry_point',
            version=getattr(dist, 'version', None), entry_point=ep)
    raw = env.get(ENV_PLUGINS) or ''
    for path in (part.strip() for part in raw.split(',')):
        if not path or path in found:
            continue
        found[path] = _Discovered(
            name=path, source='env', module_path=path,
            version=_distribution_version(path.split('.', 1)[0]))
    return [found[name] for name in sorted(found)]


def _resolve(candidate):
    """Return the zero-argument ``register`` callable for one candidate.

    An entry point resolves to the callable directly. An environment module
    path is imported and its ``register`` attribute taken, so the value written
    in :data:`ENV_PLUGINS` is the same string the entry point's left-hand side
    would be.
    """
    if candidate.entry_point is not None:
        return candidate.entry_point.load()
    module = import_module(candidate.module_path)
    register = getattr(module, 'register', None)
    if not callable(register):
        raise AttributeError(
            f'module {candidate.module_path!r} has no callable register()')
    return register


def _check_leaves(leaves, added_charts, added_exhibits, before_exhibits):
    """Return a rejection message for ``leaves``, or None when they are sound.

    Two checks, in this order because the first has the specific diagnosis.
    An exhibit name already owned by the library or by an earlier plugin is a
    silent merge under ``register_simple_exhibit``'s extend semantics, so it is
    refused by name against the pre-snapshot. Then every leaf must name a key
    the snapshot says this plugin actually added, which catches a plugin
    describing a document it forgot to register, or worse, describing somebody
    else's.
    """
    reused = [leaf.name for leaf in leaves
              if leaf.kind == 'exhibit' and leaf.name in before_exhibits]
    if reused:
        return (f'exhibit name(s) {sorted(reused)} are already registered; '
                'reusing one would extend the existing exhibit rather than '
                'declare a new one, silently merging two unrelated plugins')
    added = {'chart': added_charts, 'exhibit': added_exhibits}
    orphans = [leaf.name for leaf in leaves if leaf.name not in added[leaf.kind]]
    if orphans:
        return (f'declared leaf(s) {sorted(orphans)} name registry keys this '
                'plugin did not register')
    return None


def _run(candidate, charts_registry, exhibits_registry):
    """Run one candidate's ``register()`` and return its :class:`LoadedPlugin`."""
    before_charts = set(charts_registry)
    before_exhibits = set(exhibits_registry)
    common = dict(name=candidate.name, version=candidate.version,
                  source=candidate.source)
    try:
        leaves = list(_resolve(candidate)() or ())
    except Exception:
        return LoadedPlugin(error=traceback.format_exc(), **common)
    added_charts = tuple(sorted(set(charts_registry) - before_charts))
    added_exhibits = tuple(sorted(set(exhibits_registry) - before_exhibits))
    bad = [leaf for leaf in leaves if not isinstance(leaf, PluginLeaf)]
    if bad:
        return LoadedPlugin(
            charts=added_charts, exhibits=added_exhibits,
            error=f'register() returned {bad[0]!r}, not a PluginLeaf', **common)
    message = _check_leaves(leaves, added_charts, added_exhibits, before_exhibits)
    if message is not None:
        return LoadedPlugin(charts=added_charts, exhibits=added_exhibits,
                            error=message, **common)
    return LoadedPlugin(leaves=tuple(leaves), charts=added_charts,
                        exhibits=added_exhibits, **common)


def load(*, allow=None, env=None, force=False):
    """Discover and run the installed plugins, once.

    Idempotent: a second call returns the same list without re-running
    anything, so a host may call it defensively. Call :func:`reset_plugins`
    first, or pass ``force=True``, to run discovery again.

    Parameters
    ----------
    allow : iterable of str, optional
        An allowlist of plugin names. ``None`` (the default) admits everything
        discovered. A hosted deployment that trusts a named set threads its
        setting through here; a plugin is Python running in the server process
        with full privileges, so there is no sandbox and the allowlist is the
        only gate.
    env : mapping, optional
        Environment to read, defaulting to :data:`os.environ`. For tests.
    force : bool, default False
        Run discovery again even if a previous call already did.

    Returns
    -------
    list of LoadedPlugin
        Every candidate discovered, in plugin-name order, **including
        failures**. A caller wanting only the working ones filters on
        :attr:`LoadedPlugin.ok`.

    Notes
    -----
    ``AGGREGATE_NO_PLUGINS`` short-circuits the whole function, and the
    resulting empty list is cached like any other, so the kill switch holds for
    the life of the process.

    :mod:`aggregate.charts` and :mod:`aggregate.exhibits` are imported here at
    the point of use rather than at module scope. That keeps ``import
    aggregate`` from pulling either provisional package in, which is the
    property ``tests/test_plugins.py`` asserts and the reason this module can be
    bound on the package root at all.
    """
    global _DONE
    if _DONE and not force:
        return list(_LOADED)
    env = os.environ if env is None else env
    _LOADED.clear()
    _DONE = True
    if _truthy(env.get(ENV_NO_PLUGINS)):
        return []
    from .charts import CHARTS
    from .exhibits import EXHIBITS
    permitted = None if allow is None else set(allow)
    for candidate in _discover(env):
        if permitted is not None and candidate.name not in permitted:
            continue
        _LOADED.append(_run(candidate, CHARTS, EXHIBITS))
    return list(_LOADED)


def loaded_plugins():
    """Return the plugins :func:`load` found, failures included.

    Returns
    -------
    list of LoadedPlugin
        Empty before :func:`load` has run, which is also what a process that
        merely imported the library reports.
    """
    return list(_LOADED)


def reset_plugins():
    """Forget the load record so :func:`load` runs discovery again.

    Notes
    -----
    This clears **this module's** record only. Neither registry has a removal
    API, so whatever a plugin registered stays registered: a second
    :func:`load` of the same plugin will see its own chart name taken and
    record a duplicate-name failure. Reaching into ``CHARTS`` or ``EXHIBITS`` to
    delete a key is a test's privilege, not a supported operation.
    """
    global _DONE
    _LOADED.clear()
    _DONE = False
