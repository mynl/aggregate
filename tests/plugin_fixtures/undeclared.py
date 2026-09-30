"""A fixture plugin describing a leaf it never registered."""

from aggregate.plugins import PluginLeaf


def register():
    """Register nothing, declare something."""
    return [PluginLeaf(name='toy_phantom', kind='chart', label='Phantom')]
