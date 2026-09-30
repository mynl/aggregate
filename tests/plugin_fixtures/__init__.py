"""Fixture plugins for ``tests/test_plugins.py``.

Each module here is a miniature out-of-tree plugin: it exposes a zero-argument
``register()`` that performs its registry calls and returns its
:class:`~aggregate.plugins.PluginLeaf` list, exactly as a real
``aggregate-<thing>`` distribution's entry point does. They are reached through
``AGGREGATE_PLUGINS``, the environment on-ramp, since that route needs no
installed distribution.

Nothing here is imported by ``__init__``: importing a module must be what runs
its registrations, so the tests control when that happens.
"""
