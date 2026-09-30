"""A fixture plugin claiming an exhibit name the library already owns.

It declares the leaf without calling ``register_simple_exhibit``, on purpose:
the loader refuses by name against the pre-load snapshot, so the refusal lands
before any registration could pollute the library's ``summary`` exhibit. A real
plugin would have made the call and been refused just the same.
"""

from aggregate.plugins import PluginLeaf


def register():
    """Declare the library's ``summary`` exhibit as though it were ours."""
    return [PluginLeaf(name='summary', kind='exhibit', label='Toy summary')]
