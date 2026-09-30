Plugins (provisional)
=====================

.. warning::

   :mod:`aggregate.plugins` is **provisional** in the sense of :pep:`411`: it is **not part of the 1.0 API contract**, and its API may change in a minor release with no deprecation period. It registers into :mod:`aggregate.charts` and :mod:`aggregate.exhibits`, both provisional themselves, so a plugin inherits their freedom to change. See :doc:`3_x_API_Stability`, section :ref:`extension-surface`.

The extension surface. A third-party distribution named ``aggregate-<thing>``, with import package ``aggregate_<thing>``, contributes charts and exhibits to ``aggregate`` without a line of its code living in the library. "Lock the API down" and "allow extras" are the same design act: the **consumption** surface users code against is frozen, and this separate, explicitly unstable **extension** surface is published for plugin authors.

A plugin declares one entry point::

    [project.entry-points."aggregate.plugins"]
    relativity = "aggregate_relativity.register:register"

whose target is a zero-argument ``register()`` performing the registry calls and returning its :class:`~aggregate.plugins.PluginLeaf` list::

    from aggregate.charts import register_chart
    from aggregate.exhibits import register_simple_exhibit
    from aggregate.plugins import PluginLeaf

    def register():
        register_simple_exhibit('relativity', 'Relativity', 'relativity_df', [PnL])
        return [PluginLeaf(name='relativity', kind='exhibit', label='Relativity',
                           hint='gini_p across peel steps and distortion families',
                           why='needs an extended P&L')]

For the experiment that is not a package yet, ``AGGREGATE_PLUGINS`` names importable modules instead, comma separated, each supplying its own ``register``. Both routes are the same loader::

    AGGREGATE_PLUGINS=aggregate_relativity.register

Nothing loads on ``import aggregate``. The **host** calls :func:`~aggregate.plugins.load`, so ``build()`` stays reproducible and the test suite does not change behavior because a sibling package was synced. ``AGGREGATE_NO_PLUGINS`` suppresses discovery entirely.

Plugins live in their **own distributions**. The library carries no ``contrib`` bag: ``extensions/`` was exactly that and was deleted at ``1.0.0a12``, because experimental code inside the library becomes the library's dependency surface, its test burden and its stability question.

.. currentmodule:: aggregate.plugins

.. autosummary::

   load
   loaded_plugins
   reset_plugins
   PluginLeaf
   LoadedPlugin

.. automodule:: aggregate.plugins
   :members:
   :undoc-members:
   :show-inheritance:
